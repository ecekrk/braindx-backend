from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import time
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

app = FastAPI(title="BrainDx API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224

MODEL_PATH = os.getenv("MODEL_PATH", "beyin_tumoru_resnet50_f16.tflite")

interpreter = None
input_details = None
output_details = None


@app.on_event("startup")
async def load_model():
    global interpreter, input_details, output_details

    if not os.path.exists(MODEL_PATH):
        print(f"UYARI: Model bulunamadı -> {MODEL_PATH}")
        return

    print(f"TFLite model yükleniyor: {MODEL_PATH}")

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite ResNet50 modeli başarıyla yüklendi.")
    print("Input details:", input_details)
    print("Output details:", output_details)


@app.get("/")
def root():
    return {
        "status": "ok",
        "app": "BrainDx API",
        "model_loaded": interpreter is not None,
        "model_name": "ResNet50 TFLite Float16"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "model_name": "ResNet50 TFLite Float16"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü formatı.")

    img_array = np.array(img, dtype=np.float32)

    # ResNet50 eğitim ön işlemesiyle uyumlu
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    t0 = time.time()

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    elapsed_ms = int((time.time() - t0) * 1000)

    predicted_idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(preds[predicted_idx])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, preds)
        },
        "inference_time_ms": elapsed_ms,
        "model_name": "ResNet50 TFLite Float16"
    }