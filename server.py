from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from datetime import datetime
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# --- KAN Layer with optional activation ---
class KANLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, activation='gelu'):
        super(KANLayer, self).__init__()
        self.weight = self.add_weight(
            shape=(output_dim, input_dim),
            initializer="he_normal",
            trainable=True,
            name="kan_weights"
        )
        self.bias = self.add_weight(
            shape=(output_dim,),
            initializer="zeros",
            trainable=True,
            name="kan_bias"
        )
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        x = tf.tensordot(inputs, self.weight, axes=1) + self.bias
        return self.activation(x)


# --- Fast Attention with residual ---
class FastAttentionLayer(layers.Layer):
    def __init__(self, output_dim):
        super(FastAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.query_proj = layers.Dense(output_dim)
        self.key_proj = layers.Dense(output_dim)
        self.value_proj = layers.Dense(output_dim)

    def call(self, inputs):
        input_rank = inputs.shape.rank
        if input_rank == 4:
            b, h, w, c = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
            n = h * w
            x = tf.reshape(inputs, [b, n, c])
            Q = self.query_proj(tf.nn.l2_normalize(x, axis=-1))
            K = self.key_proj(tf.nn.l2_normalize(x, axis=-1))
            V = self.value_proj(x)
            KV = tf.matmul(K, V, transpose_a=True)
            Y = tf.matmul(Q, KV) / tf.cast(n, tf.float32)
            return tf.reshape(Y + x, [b, h, w, self.output_dim])
        elif input_rank == 3:
            n = tf.shape(inputs)[1]
            Q = self.query_proj(tf.nn.l2_normalize(inputs, axis=-1))
            K = self.key_proj(tf.nn.l2_normalize(inputs, axis=-1))
            V = self.value_proj(inputs)
            KV = tf.matmul(K, V, transpose_a=True)
            Y = tf.matmul(Q, KV) / tf.cast(n, tf.float32)
            return Y + inputs
        else:
            raise ValueError("Unsupported input rank.")


# --- Tokenized KAN Block with stacking ---
def tokenized_kan_block_student(inputs, token_dim, kan_layers=2):
    tokens = layers.Reshape((-1, inputs.shape[-1]))(inputs)
    tokens = layers.Dense(token_dim, activation='relu')(tokens)

    x = tokens
    for _ in range(kan_layers):
        y = KANLayer(token_dim, token_dim)(x)
        y = layers.LayerNormalization()(y)
        x = layers.Add()([x, y])

    x = FastAttentionLayer(token_dim)(x)
    x = layers.LayerNormalization()(x)

    # Use Conv2D to project the input to the same dimension
    projected = layers.Conv2D(token_dim, (1, 1), padding='same', activation='relu')(inputs)

    # Use Lambda to dynamically reshape `x` to match the shape of `projected`
    x_reshaped = layers.Lambda(lambda x: tf.reshape(x, (-1, projected.shape[1], projected.shape[2], token_dim)))(x)

    # Perform Add operation
    tokens = layers.Add()([x_reshaped, projected])

    out = KANLayer(token_dim, token_dim)(tokens)
    out = layers.LayerNormalization()(out)

    return layers.Reshape((inputs.shape[1], inputs.shape[2], token_dim))(out)


# --- Fuse and Up ---
# --- Fuse and Up (updated) ---
def fuse_up(skip, up_input, out_channels):
    upsampled = layers.UpSampling2D((2, 2), interpolation='bilinear')(up_input)
    height, width = upsampled.shape[1], upsampled.shape[2]
    skip_resized = layers.Resizing(height, width, interpolation='bilinear')(skip)

    # Align channel dimensions before Add
    if skip_resized.shape[-1] != upsampled.shape[-1]:
        skip_resized = layers.Conv2D(upsampled.shape[-1], (1, 1), padding='same', use_bias=False)(skip_resized)

    # skip_resized = se_block(skip_resized)

    x = layers.Add()([upsampled, skip_resized])
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)(x)
    return x


# --- Student Model Building Function ---
def build_student_model(input_shape, kan_dim=64, num_kan_layers=1):  # Giảm `kan_dim` và `num_kan_layers`
    inputs = layers.Input(shape=input_shape)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Fine-tune từ block_13 trở đi
    for layer in base_model.layers:
        if 'block_13' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    c1 = base_model.get_layer('block_1_expand_relu').output
    c2 = base_model.get_layer('block_3_expand_relu').output
    c3 = base_model.get_layer('block_6_expand_relu').output
    c4 = base_model.get_layer('block_13_expand_relu').output

    bottleneck = tokenized_kan_block_student(c4, kan_dim, num_kan_layers)

    c4_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c4)  # Giảm số lượng filters
    c4_skip = FastAttentionLayer(64)(c4_skip)  # Giảm số lượng FastAttentionLayer
    u1 = fuse_up(c4_skip, bottleneck, 32)  # Giảm số lượng filters

    c3_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c3)
    c3_skip = FastAttentionLayer(64)(c3_skip)
    u2 = fuse_up(c3_skip, u1, 16)  # Giảm số lượng filters

    c2_skip = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(c2)
    c2_skip = FastAttentionLayer(32)(c2_skip)
    u3 = fuse_up(c2_skip, u2, 8)  # Giảm số lượng filters

    c1_skip = layers.Conv2D(16, (1, 1), padding='same', use_bias=False)(c1)
    # c1_skip = FastAttentionLayer(16)(c1_skip)
    u4 = fuse_up(c1_skip, u3, 8)  # Giảm số lượng filters

    u4 = layers.Dropout(0.05)(u4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


# Instantiate the student model with optimized parameters
input_shape = (288, 288, 3)  # Bạn có thể giảm kích thước đầu vào ở đây
model = build_student_model(input_shape, kan_dim=16, num_kan_layers=2)  # Giảm kan_dim và num_kan_layers

try:
    model.load_weights('distilled_student_model_weights.weights.h5')
    print("Weights loaded successfully")
except ValueError as e:
    print(f"Error loading weights: {e}")

# ✅ Global variable to store the latest detection result
latest_result = {
    "fire_detected": False,
    "timestamp": None,
    "image_url": None,
    "bounding_box": None  # Giữ lại nếu frontend vẫn expect field này
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

import uuid
import os

@app.post("/predict/")
async def predict_fire(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        frame_resized = cv2.resize(frame, (288, 288))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = frame_rgb.astype("float32") / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        prediction = model.predict(input_tensor)
        fire_mask = prediction[0, :, :, 0] > 0.5  # Extract channel 0 explicitly
        fire_detected = bool(prediction.max() > 0.5)

        alert_image_url = None
        if fire_detected:
            # Tạo mask màu đỏ (BGR)
            fire_mask_uint8 = fire_mask.astype(np.uint8) * 255
            mask_color = cv2.merge([
                fire_mask_uint8,              # Blue channel
                np.zeros_like(fire_mask_uint8),  # Green channel
                np.zeros_like(fire_mask_uint8)   # Red channel
            ])  # Red fire mask

            # Overlay mask lên ảnh gốc
            frame_overlayed = cv2.addWeighted(frame_resized, 1.0, mask_color, 0.5, 0)

            # Lưu ảnh đã overlay
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join("static", "alerts", filename)
            cv2.imwrite(filepath, frame_overlayed)
            alert_image_url = f"/static/alerts/{filename}"

        if fire_detected:
            latest_result.update({
                "fire_detected": fire_detected,
                "timestamp": datetime.utcnow().isoformat(),
                "image_url": alert_image_url,
                "bounding_box": None  # Không dùng bounding box nữa
            })

        return {
            "fire_detected": fire_detected,
            "image_url": alert_image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/latest-alert/")
def get_latest_alert():
    # ✅ Return the most recent prediction result
    return latest_result
