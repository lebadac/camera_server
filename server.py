from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

app = FastAPI()

# ✅ Load model once at server startup
try:
    model = load_model("fire_segmentation_model.h5", compile=False)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# ✅ Global variable to store the latest detection result
latest_result = {
    "fire_detected": False,
    "timestamp": None,
    "bounding_box": None
}

@app.get("/")
def read_root():
    return {"message": "🔥 Fire Detection Server is running!"}

# ✅ Helper function to calculate bounding box from binary mask
def get_bounding_box(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return [int(x), int(y), int(x + w), int(y + h)]  # Format: [x1, y1, x2, y2]

@app.post("/predict/")
async def predict_fire(file: UploadFile = File(...)):
    try:
        # ✅ Read uploaded image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # ✅ Preprocess image
        frame_resized = cv2.resize(frame, (288, 288))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = frame_rgb.astype("float32") / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # ✅ Run model prediction
        prediction = model.predict(input_tensor)
        fire_mask = prediction[0] > 0.5

        # ✅ Determine fire presence
        fire_detected = bool(prediction.max() > 0.5)
        bbox = get_bounding_box(fire_mask) if fire_detected else None

        # ✅ Update latest global result
        latest_result.update({
            "fire_detected": fire_detected,
            "timestamp": datetime.utcnow().isoformat(),
            "bounding_box": bbox
        })

        # ✅ Return response to ESP32
        return {
            "fire_detected": fire_detected,
            "bounding_box": bbox,
            "width": fire_mask.shape[1],
            "height": fire_mask.shape[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/latest-alert/")
def get_latest_alert():
    # ✅ Return the most recent prediction result
    return latest_result
