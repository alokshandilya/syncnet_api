import os
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile

from syncnet import analyze_sync, load_syncnet_model

app = FastAPI(title="CPU Lip Sync Detector")

# Global Load of Model (Load once on startup)
model = None


@app.on_event("startup")
def startup_event():
    global model
    print("Loading SyncNet model...")
    try:
        model = load_syncnet_model("syncnet_v2.model")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.post("/check-sync")
async def check_sync_endpoint(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Save uploaded file to temp
    file_id = str(uuid.uuid4())
    temp_filename = f"temp_{file_id}.mp4"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Run Analysis
        result = analyze_sync(temp_filename, model)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 3. Cleanup: Delete temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.get("/")
def home():
    return {"message": "SyncNet API is running. POST video to /check-sync"}
