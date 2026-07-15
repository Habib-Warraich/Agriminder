import os
import io
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# --- 1. GEMINI AI CONFIG ---
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
vision_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. CNN MODEL LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

try:
    # compile=False is necessary for Render's memory limit
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ CNN Brain Active on Cloud")
except Exception as e:
    model = None
    print(f"❌ Error: {e}")

CLASSES = ['Apple Brown_spot', 'Apple Normal', 'Apple black_spot', 'Apricot Normal', 'Apricot blight leaf disease', 'Apricot shot_hole', 'Bean Fungal_leaf disease', 'Bean Normal leaf', 'Bean bean rust image', 'Bean shot_hole', 'Cherry Leaf Scorch', 'Cherry Normal leaf', 'Cherry brown_spot', 'Cherry purple leaf spot', 'Cherry_shot hole disease', 'Corn Fungal leaf', 'Corn Normal leaf', 'Corn gray leaf spot', 'Corn holcus_ leaf spot', 'Fig Blight_leaf disease', 'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf', 'Grape Anthracnose leaf', 'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease', 'Lokat Normal leaf', 'Pear Black spot _ leaf disease', 'Pear Normal _leaf', 'Pear fire blight', 'Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 'Walnut Normal_leaf', 'Walnut Shot_hole', 'Walnut leaf gall mite', 'lokat Leaf_spot', 'persimmons Brown_spot', 'tomato Fusarium Wilt', 'tomato spider mites', 'tomato verticillium wilt', 'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy_leaf', 'tomato_late_blight', 'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf']

def get_ai_analysis(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        prompt = "Analyze this image. If it is a plant leaf, identify it and any disease. Tell Cause and Cure in detail for a Pakistani farmer. If it is NOT a plant leaf, reply ONLY with 'INVALID'."
        response = vision_model.generate_content([prompt, img])
        return response.text
    except: return "AI analysis error."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    img_bytes = file.read()
    
    # 1. Universal AI Check (Bypass Laptop Issue)
    ai_detail = get_ai_analysis(img_bytes)
    if "INVALID" in ai_detail:
        return jsonify({"disease": "Object Not Recognized", "status": "invalid", "urdu": "اے آئی اسے پودا تسلیم نہیں کر رہی۔"})

    # 2. CNN Precise Confidence
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
    img_array = preprocess_input(np.array(img).astype('float32'))
    img_array = np.expand_dims(img_array, axis=0)

    if model:
        preds = model.predict(img_array)
        confidence = f"{round(np.max(preds[0]) * 100, 1)}%"
        disease = CLASSES[np.argmax(preds[0])]
        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "details": ai_detail,
            "status": "healthy" if "Normal" in disease else "danger"
        })
    return jsonify({"error": "Model Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)