import os
import shutil
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile

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
async def check_sync_endpoint(request: Request):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        file_path = body.get("file_path")
        if not file_path:
            raise HTTPException(status_code=400, detail="JSON payload must contain 'file_path'")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

        try:
            # Run Analysis directly on the path
            result = analyze_sync(file_path, model)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid form data")

        file = form.get("file")
        if not file:
            raise HTTPException(status_code=400, detail="Form data must contain 'file'")

        # Check if it is really an upload
        if not isinstance(file, UploadFile):
            raise HTTPException(status_code=400, detail="Field 'file' is not a valid file upload")

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

    else:
        raise HTTPException(status_code=400, detail="Content-Type must be application/json or multipart/form-data")


@app.get("/")
def home():
    return {"message": "SyncNet API is running. POST video to /check-sync"}