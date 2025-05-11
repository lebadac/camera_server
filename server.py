from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

app = FastAPI()

# ✅ Load model once (tối ưu hơn)
try:
    model = load_model("fire_segmentation_model.h5", compile=False)
except Exception as e:
    raise RuntimeError(f"Lỗi khi load model: {str(e)}")

@app.get('/')
def read_root():
    return {"message": "Deep Learning model deployed"}

@app.post("/predict/")
async def predict_fire(file: UploadFile = File(...)):
    try:
        # ✅ Đọc file ảnh
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Lỗi đọc ảnh! Kiểm tra lại file upload.")

        # ✅ Tiền xử lý ảnh
        frame_resized = cv2.resize(frame, (288, 288))
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_resized = frame_resized.astype("float32") / 255.0
        frame_resized = np.expand_dims(frame_resized, axis=0)

        # ✅ Dự đoán
        prediction = model.predict(frame_resized)

        # ✅ Xác định có cháy hay không
        fire_detected = bool(prediction.max() > 0.5)  # Cách tối ưu hơn np.any()

        fire_mask = prediction[0] > 0.5  # Binary mask (0 or 1)
        fire_mask_flat = fire_mask.astype(np.uint8).flatten().tolist()  # Convert to 1D list

        return {
            "fire_detected": fire_detected,
            "fire_mask": fire_mask_flat,
            "width": fire_mask.shape[1],  # Send width
            "height": fire_mask.shape[0]  # Send height
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
