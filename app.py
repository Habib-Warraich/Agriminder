import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# TENSORFLOW 2.16+ AUR KERAS 3 FIX
import keras
from keras.applications.mobilenet_v2 import preprocess_input

# Environment Fixes
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

app = Flask(__name__)
CORS(app)

# --- 1. GEMINI AI CONFIG ---
# Aapki API Key
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
ai_engine = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. CNN MODEL LOADING (Absolute Path Fix) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

model = None
try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ CNN Brain Loaded Successfully")
    else:
        print(f"❌ Error: model.h5 not found at {model_path}")
except Exception as e:
    print(f"❌ Load Error: {e}")

CLASSES = ['Apple Brown_spot', 'Apple Normal', 'Apple black_spot', 'Apricot Normal', 'Apricot blight leaf disease', 'Apricot shot_hole', 'Bean Fungal_leaf disease', 'Bean Normal leaf', 'Bean bean rust image', 'Bean shot_hole', 'Cherry Leaf Scorch', 'Cherry Normal leaf', 'Cherry brown_spot', 'Cherry purple leaf spot', 'Cherry_shot hole disease', 'Corn Fungal leaf', 'Corn Normal leaf', 'Corn gray leaf spot', 'Corn holcus_ leaf spot', 'Fig Blight_leaf disease', 'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf', 'Grape Anthracnose leaf', 'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease', 'Lokat Normal leaf', 'Pear Black spot _ leaf disease', 'Pear Normal _leaf', 'Pear fire blight', 'Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 'Walnut Normal_leaf', 'Walnut Shot_hole', 'Walnut leaf gall mite', 'lokat Leaf_spot', 'persimmons Brown_spot', 'tomato Fusarium Wilt', 'tomato spider mites', 'tomato verticillium wilt', 'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy_leaf', 'tomato_late_blight', 'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf']

def get_ai_analysis(img_bytes, cnn_result):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        prompt = f"""
        Analyze this plant leaf. CNN says: {cnn_result}.
        1. If NOT a plant (laptop/person), reply ONLY 'INVALID'.
        2. If a leaf, identify disease, Cause (water/NPK/humidity) and Cure.
        3. Short summary in Urdu.
        """
        response = ai_engine.generate_content([prompt, img])
        return response.text
    except: return "Analysis currently unavailable."

@app.route('/')
def home():
    return "AgriMinder Backend is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    try:
        file = request.files['file']
        img_bytes = file.read()
        
        # 1. Run CNN
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        img_array = preprocess_input(np.array(img).astype('float32'))
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            preds = model.predict(img_array)
            confidence = np.max(preds[0])
            cnn_disease = CLASSES[np.argmax(preds[0])]

            # 2. AI Analysis
            ai_details = get_ai_analysis(img_bytes, cnn_disease)

            if "INVALID" in ai_details.upper() and confidence < 0.50:
                return jsonify({"disease": "Object Not Recognized", "status": "invalid", "urdu": "پودا نہیں ملا۔"})

            return jsonify({
                "disease": cnn_disease,
                "confidence": f"{round(confidence * 100, 1)}%",
                "details": ai_advice if (ai_advice := ai_details) else "Processing...",
                "status": "healthy" if "Normal" in cnn_disease else "danger"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
