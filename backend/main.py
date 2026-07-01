from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("ringworm_binary_classifier.onnx")
input_name = session.get_inputs()[0].name

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    input_data = preprocess(image)
    result = session.run(None, {input_name: input_data})[0][0][0]
    confidence = float(result)
    has_ringworm = confidence > 0.5
    return {
        "ringworm": has_ringworm,
        "confidence": round(confidence if has_ringworm else 1 - confidence, 3)
    }

@app.get("/")
def root():
    return {"status": "RingCheck API running"}