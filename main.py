import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from mtcnn import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64

# ── 1. CONFIGURATION & MODELS ────────────────────────────────────────────────
emotion_labels_8_classes = [
    'Angry', 'Disgust', 'Fear', 'Happy', 
    'Sad', 'Surprise', 'Neutral', 'Contempt'
]

print("Initializing models... please wait.")
try:
    # FER model (Expected input: features from ResNet50)
    model = load_model("fermodel.keras")
    
    # ResNet50 for feature extraction
    resnet50 = ResNet50(
        weights="imagenet", 
        include_top=False, 
        pooling="avg", 
        input_shape=(224, 224, 3)
    )
    
    # Face Detector
    detector = MTCNN()
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# ── 2. FASTAPI SETUP WITH CORS ───────────────────────────────────────────────
app = FastAPI(title="Emotion Recognition & Face Detection API")

# This block fixes the "Access-Control-Allow-Origin" error in React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ── 3. UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def get_image_from_bytes(contents):
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image_to_base64(image_array):
    """Converts OpenCV image to a Base64 string for JSON transfer."""
    _, buffer = cv2.imencode('.jpg', image_array)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

# ── 4. ENDPOINT 1: PREDICT EMOTION ───────────────────────────────────────────

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = get_image_from_bytes(contents)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect Face
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        
        if not results:
            return {"status": "failure", "message": "No face detected"}

        x, y, w, h = results[0]['box']
        face = img[max(0, y):y+h, max(0, x):x+w]
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (48, 48))
        img_224 = cv2.resize(gray_resized, (224, 224))
        img_rgb_input = cv2.cvtColor(img_224, cv2.COLOR_GRAY2RGB)
        

        prep = preprocess_input(np.expand_dims(img_rgb_input.astype("float32"), axis=0))
        features = resnet50.predict(prep, verbose=0).reshape(1, 1, -1)
        preds = model.predict(features, verbose=0)[0]
        class_idx = np.argmax(preds)
        return {
            "status": "success",
            "emotion": emotion_labels_8_classes[class_idx],
            "confidence": round(float(preds[class_idx]), 4),
            "probabilities": {
                emotion_labels_8_classes[i]: round(float(preds[i]), 4) 
                for i in range(len(emotion_labels_8_classes))
            },
            "message": f"{emotion_labels_8_classes[class_idx]} detected."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 5. ENDPOINT 2: DETECT FACE (CROPPED IMAGE + COORDS) ──────────────────────

@app.post("/detect-face")
async def detect_face_only(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = get_image_from_bytes(contents)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        
        if not results:
            return {"face_found": False, "message": "No face detected"}

        x, y, w, h = results[0]['box']
        cropped_face = img[max(0, y):y+h, max(0, x):x+w]
        
        return {
            "face_found": True,
            "coordinates": {"x": x, "y": y, "width": w, "height": h},
            "image_data": encode_image_to_base64(cropped_face)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 5. RUN LOGIC ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        print("Starting FastAPI Server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Starting Webcam Mode. (To run API, use: python filename.py api)")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret: break

            res = predict_emotion_wrapper(frame)
            if res["message"] != "No face detected" and res["emotion"] != "N/A":
                cv2.putText(frame, res["message"], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()