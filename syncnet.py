import torch
import torch.nn as nn
import cv2
import numpy as np
import librosa
import os
import shutil
import subprocess
import tempfile

# Constants
SYNC_THRESHOLD = 7.5

# --- A. Define SyncNet Architecture ---
class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()
        
        # Audio Encoder - CNN Part
        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            nn.Conv2d(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            nn.Conv2d(192, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            # Final Conv to collapse valid region to 1x1
            nn.Conv2d(256, 512, kernel_size=(5,4), padding=(0,0)),
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        
        # Audio Encoder - FC Part (Linear layers)
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024),
        )
        
        # Video Encoder - CNN Part
        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            nn.Conv3d(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            
            # Note: A MaxPool3d layer is normally present here in SyncNet V2, 
            # but we omit it to support 111x111 inputs directly without 
            # downsampling the feature map below the 6x6 kernel size of the next layer.
            
            nn.Conv3d(256, 512, kernel_size=(1,6,6), padding=0),
            nn.BatchNorm3d(512), nn.ReLU(),
        )
        
        # Video Encoder - FC Part (Linear layers)
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024),
        )

    def forward(self, audio, video):
        # Audio Forward
        audio = self.netcnnaud(audio)
        audio = audio.view(audio.size(0), -1) # Flatten (B, 512, 1, 1) -> (B, 512)
        audio = self.netfcaud(audio)

        # Video Forward
        video = self.netcnnlip(video)
        video = video.view(video.size(0), -1) # Flatten (B, 512, 1, 1, 1) -> (B, 512)
        video = self.netfclip(video)
        
        return audio, video

# --- B. Helper: Load Model ---
def load_syncnet_model(model_path="syncnet_v2.model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}")

    model = SyncNet()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    # Fix dict keys if 'module.' prefix exists
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    
    # Patch keys to account for removed MaxPool3d layer in netcnnlip
    # The checkpoint expects layers at indices 18 and 19 (because 17 was a MaxPool),
    # but our modified sequential block has them at 17 and 18.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("netcnnlip.18."):
            # Shift Conv weights from 18 to 17
            new_k = k.replace("netcnnlip.18.", "netcnnlip.17.")
            new_state_dict[new_k] = v
        elif k.startswith("netcnnlip.19."):
            # Shift BN weights from 19 to 18
            new_k = k.replace("netcnnlip.19.", "netcnnlip.18.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
            
    # Try strict loading; print error but proceed if minor mismatch to allow debugging
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Warning: strict loading failed ({e}), trying strict=False")
        model.load_state_dict(new_state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model

# --- C. Core Logic: Process Video ---
def analyze_sync(video_path, model):
    # Determine device from model parameters
    device = next(model.parameters()).device

    # 1. Init Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 2. Load Audio & Video
    if not shutil.which("ffmpeg"):
        return {"error": "Audio processing failed: FFmpeg is not installed or not in PATH."}

    # Librosa load (resample to 16k)
    # Use ffmpeg directly to extract wav to avoid NoBackendError with audioread
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio_path = tmp.name

        # Extract audio: mono, 16kHz
        process = subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-vn", "-ac", "1", "-ar", "16000", 
            temp_audio_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if process.returncode != 0:
            error_msg = process.stderr.decode('utf-8', errors='ignore')
            # Handle case where video has no audio
            if "Output file does not contain any stream" in error_msg:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return {"error": "Video file has no audio stream."}
            
            raise RuntimeError(f"FFmpeg error (code {process.returncode}): {error_msg}")

        audio_full, sr = librosa.load(temp_audio_path, sr=16000)
        
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    except Exception as e:
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return {"error": f"Audio processing failed: {type(e).__name__}: {str(e)}"}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: return {"error": "Could not read video FPS"}

    frame_buffer = []
    sync_scores = []
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- Face Detection (Downscale for speed) ---
        h, w = frame.shape[:2]
        
        # Resize while maintaining aspect ratio to avoid distortion
        target_w = 480
        scale_factor = target_w / float(w)
        new_h = int(h * scale_factor)
        
        small_frame = cv2.resize(frame, (target_w, new_h))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

        if len(faces) > 0:
            # Get largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            (sx, sy, sw, sh) = faces[0]
            
            x = int(sx / scale_factor)
            y = int(sy / scale_factor)
            bw = int(sw / scale_factor)
            bh = int(sh / scale_factor)

            # Approximating mouth region (bottom half of face)
            mouth_y = y + int(bh * 0.5)
            mouth_h = int(bh * 0.5)
            
            # Crop
            if mouth_y < h and x+bw < w and x > 0:
                crop = frame[mouth_y:mouth_y+mouth_h, x:x+bw]
                try:
                    crop = cv2.resize(crop, (111, 111))
                    frame_buffer.append(crop)
                except:
                    pass # resize failed
        else:
            # If no face, clear buffer to avoid mixing non-consecutive frames
            frame_buffer = []

        # --- Inference Trigger (Every 5 frames) ---
        # We need 5 consecutive frames for video input
        if len(frame_buffer) >= 5:
            # Create Video Tensor: [1, 3, 5, 111, 111]
            vid_batch = np.stack(frame_buffer[-5:], axis=0)
            vid_batch = np.transpose(vid_batch, (3, 0, 1, 2)) # HWC -> CHW
            vid_tensor = torch.FloatTensor(vid_batch).unsqueeze(0).to(device)

            # Audio extraction logic matching the current 5 frames
            # Center time of the 5-frame window
            curr_time_sec = frame_idx / fps
            center_idx = int(curr_time_sec * 16000)
            
            # SyncNet expects ~0.2s audio context. 
            # We take a window around the current frame time.
            start_idx = center_idx - 3200  # -0.2s
            end_idx = center_idx + 3200    # +0.2s (Generous window to crop MFCC)
            
            if start_idx >= 0 and end_idx < len(audio_full):
                audio_chunk = audio_full[start_idx:end_idx]
                
                # Create MFCC
                mfcc = librosa.feature.mfcc(y=audio_chunk, sr=16000, n_mfcc=13, hop_length=160, win_length=400, center=False)
                # We need exactly 20 time steps. 
                if mfcc.shape[1] >= 20:
                    mfcc = mfcc[:, :20] # trim
                    aud_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        a_emb, v_emb = model(aud_tensor, vid_tensor)
                        # Flatten embeddings before distance calc, though usually they are 1x1
                        dist = torch.dist(a_emb, v_emb, 2).item()
                        sync_scores.append(dist)

            # Slide window: remove oldest frame
            frame_buffer.pop(0)

        frame_idx += 1

    cap.release()

    if not sync_scores:
        return {"status": "failed", "reason": "No faces or sync detected"}

    avg_dist = np.mean(sync_scores)
    # Threshold: < SYNC_THRESHOLD is "Good", > SYNC_THRESHOLD is "Bad"
    is_good = avg_dist < SYNC_THRESHOLD

    return {
        "status": "success",
        "average_distance": float(avg_dist),
        "is_sync_good": bool(is_good),
        "frames_analyzed": len(sync_scores)
    }