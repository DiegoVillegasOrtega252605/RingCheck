import os
import io
import time
import uuid

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import cloudinary
import cloudinary.uploader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("ringworm_binary_classifier.onnx")
input_name = session.get_inputs()[0].name

# --- Cloudinary config (reads from environment variables set on Render) ---
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True,
)

ALLOWED_LABELS = {"ringworm", "healthy", "benign_mark", "unsure"}


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


@app.post("/contribute")
async def contribute(
    file: UploadFile = File(...),
    label: str = Form(...),
    consent: bool = Form(...),
    is_adult: bool = Form(...),
):
    """
    Public community submission endpoint. Accepts a photo + self-reported label
    (ringworm / healthy / unsure) from external contributors (e.g. via a social
    media call-out), for manual review before being added to the training set.
    Requires explicit consent and an 18+ confirmation.
    """
    if label not in ALLOWED_LABELS:
        return {"error": f"label must be one of {sorted(ALLOWED_LABELS)}"}

    if not consent or not is_adult:
        return {"error": "Consent and 18+ confirmation are required to submit."}

    contents = await file.read()

    # Basic validation: make sure it's actually a readable image
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:
        return {"error": "File does not appear to be a valid image."}

    unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"

    try:
        upload_result = cloudinary.uploader.upload(
            io.BytesIO(contents),
            folder=f"ringcheck-contributions/{label}",
            public_id=unique_id,
            resource_type="image",
        )
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

    return {
        "status": "received",
        "message": "Thank you for contributing! Your photo will be reviewed before use.",
        "id": unique_id,
        "url": upload_result.get("secure_url"),
    }


@app.get("/")
def root():
    return {"status": "RingCheck API running"}
