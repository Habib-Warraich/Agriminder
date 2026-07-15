import os
import io
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# MobileNetV2 preprocessing logic
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# --- 1. GEMINI AI CONFIGURATION ---
GENIMINI_KEY = "AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA"
genai.configure(api_key=GENIMINI_KEY)
ai_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. CNN MODEL LOADING (Legacy Fix) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

model = None
try:
    if os.path.exists(model_path):
        # compile=False enables loading even if there are version mismatches
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ CNN Brain Loaded Successfully (Legacy Environment)")
    else:
        print(f"❌ Error: {model_path} not found")
except Exception as e:
    print(f"❌ Critical Load Error: {e}")

# 52 Classes (Alphabetical order)
CLASSES = [
    'Apple Brown_spot', 'Apple Normal', 'Apple black_spot', 'Apricot Normal', 
    'Apricot blight leaf disease', 'Apricot shot_hole', 'Bean Fungal_leaf disease', 
    'Bean Normal leaf', 'Bean bean rust image', 'Bean shot_hole', 'Cherry Leaf Scorch', 
    'Cherry Normal leaf', 'Cherry brown_spot', 'Cherry purple leaf spot', 
    'Cherry_shot hole disease', 'Corn Fungal leaf', 'Corn Normal leaf', 
    'Corn gray leaf spot', 'Corn holcus_ leaf spot', 'Fig Blight_leaf disease', 
    'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf', 'Grape Anthracnose leaf', 
    'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 
    'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease', 
    'Lokat Normal leaf', 'Pear Black spot _ leaf disease', 'Pear Normal _leaf', 
    'Pear fire blight', 'Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 
    'Walnut Normal_leaf', 'Walnut Shot_hole', 'Walnut leaf gall mite', 
    'lokat Leaf_spot', 'persimmons Brown_spot', 'tomato Fusarium Wilt', 
    'tomato spider mites', 'tomato verticillium wilt', 'tomato_bacterial_spot', 
    'tomato_early_blight', 'tomato_healthy_leaf', 'tomato_late_blight', 
    'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf'
]

def get_ai_advice(disease_name):
    """Generative AI to explain Cause and Cure"""
    prompt = f"""
    Analyze this plant disease detected by my CNN: {disease_name}.
    Explain:
    1. Primary Cause: (e.g., poor water system, nitrogen deficiency, humidity).
    2. Cure: Exact pesticide/fertilizer name or organic method.
    Keep it short and professional for a farmer in Pakistan.
    Add a 1-line summary in Urdu at the end.
    """
    try:
        response = ai_model.generate_content(prompt)
        return response.text
    except:
        return "Cause: Humidity issues. Cure: Apply fungicide and consult expert."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img).astype('float32')
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            predictions = model.predict(img_array)
            confidence = np.max(predictions[0])
            index = np.argmax(predictions[0])

            # Accuracy Guard (Laptop detection fix)
            if confidence < 0.75:
                return jsonify({
                    "disease": "Object Not Recognized",
                    "confidence": "N/A",
                    "details": "This is not a plant leaf. Please scan a clear leaf image.",
                    "urdu": "اے آئی اسے پودا تسلیم نہیں کر رہی۔"
                })

            disease = CLASSES[index]
            # Get insights from Gemini
            ai_details = get_ai_advice(disease)

            return jsonify({
                "disease": disease,
                "confidence": f"{round(confidence * 100, 1)}%",
                "details": ai_details,
                "status": "danger" if "Normal" not in disease else "healthy"
            })
        
        return jsonify({"error": "CNN model failed to load on server"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return "AgriMinder AI API is Running"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
