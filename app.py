import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Keras 3 compatibility logic
import keras
from keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# --- 1. MODEL LOADING (FIXED FOR KERAS 3) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

try:
    if os.path.exists(model_path):
        # H5 file ke liye 'compile=False' hi sab se best fix hai
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ CNN Brain Loaded Successfully")
    else:
        model = None
        print(f"⚠️ model.h5 not found at {model_path}")
except Exception as e:
    model = None
    print(f"❌ Load Error: {e}")

# ... (Keep your CLASSES list and routes exactly the same as before)

# IMPORTANT: This list MUST be in Alphabetical Order 
# to match how Keras trained your 52 classes.
# Run 'print(sorted(os.listdir("dataset/train")))' in your training script to verify.
CLASSES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Gray Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Orange Haunglongbing", "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy", "Potato Early Blight",
    "Potato Late Blight", "Potato Healthy", "Raspberry Healthy", "Soybean Healthy",
    "Squash Powdery Mildew", "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy",
    "Wheat Yellow Rust", "Wheat Brown Rust", "Wheat Healthy" 
    # Add your remaining classes here in alphabetical order
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224)) # MobileNetV2 Input Size
        
        if model:
            # --- PRECISION PREPROCESSING ---
            img_array = np.array(img).astype('float32')
            # This scales pixels to [-1, 1] exactly like the training
            img_array = preprocess_input(img_array) 
            img_array = np.expand_dims(img_array, axis=0)

            # --- INFERENCE ---
            predictions = model.predict(img_array)
            confidence = np.max(predictions[0])
            index = np.argmax(predictions[0])

            # --- VALIDATION LAYER (SOLVES THE LAPTOP ISSUE) ---
            # If the AI is not at least 75% sure, it's probably not a leaf
            if confidence < 0.75:
                return jsonify({
                    "disease": "Object Not Recognized",
                    "confidence": f"{round(confidence * 100, 2)}%",
                    "treatment": "Please scan a clear plant leaf. AI cannot verify this object.",
                    "urdu": "اے آئی اس چیز کو نہیں پہچان سکی۔ براہ کرم پتے کی صاف تصویر لیں۔"
                })

            disease = CLASSES[index] if index < len(CLASSES) else "New Disease Type"
            
            return jsonify({
                "disease": disease,
                "confidence": f"{round(confidence * 100, 2)}%",
                "treatment": "Apply Propiconazole (250ml/acre) for rust. Check NPK levels.",
                "urdu": f"تشخیص: {disease}۔ متاثرہ حصے پر اسپرے کریں۔"
            })
        else:
            return jsonify({"disease": "Demo Mode: Healthy", "confidence": "99%"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Processing failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
