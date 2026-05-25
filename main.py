from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io, time, os
import tensorflow as tf

app = FastAPI(title="BrainDx API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Sınıf etiketleri ─────────────────────────────────────────────
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE    = 224
model       = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "brain_tumor_model.h5")
    if os.path.exists(model_path):
        print(f"Model yükleniyor: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model başarıyla yüklendi!")
    else:
        print(f"UYARI: Model bulunamadı → {model_path}")

@app.get("/")
def root():
    return {"status": "ok", "app": "BrainDx API", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    # Görüntüyü oku ve işle
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü formatı")

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    t0 = time.time()
    preds = model.predict(img_array, verbose=0)[0]
    elapsed_ms = int((time.time() - t0) * 1000)

    predicted_idx   = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "confidence":      float(preds[predicted_idx]),
        "probabilities":   {name: float(prob) for name, prob in zip(CLASS_NAMES, preds)},
        "inference_time_ms": elapsed_ms,
        "model_name":      "EfficientNetB3"
    }
