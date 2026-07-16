import os
import io
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# --- THE MAGIC HACK (STOPS THE DENSE LAYER ERROR) ---
# Yeh hissa load_model se pehle hona chahiye
from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer

def patch_layer(layer_class):
    original_init = layer_class.__init__
    def patched_init(self, *args, **kwargs):
        # In keys ko nikaal do jo error deti hain
        kwargs.pop('quantization_config', None)
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        kwargs.pop('synchronized', None)
        original_init(self, *args, **kwargs)
    layer_class.__init__ = patched_init

# In saari layers ko theek karo
patch_layer(Dense)
patch_layer(BatchNormalization)
# ---------------------------------------------------

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

app = Flask(__name__)
CORS(app)

# --- 1. AI CONFIGURATION ---
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
ai_engine = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. CNN MODEL LOADING ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

try:
    if os.path.exists(model_path):
        # compile=False zaroori hai version mismatch ke liye
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ CNN Brain Loaded Successfully with Patch")
    else:
        model = None
        print(f"❌ Error: model.h5 not found at {model_path}")
except Exception as e:
    model = None
    print(f"❌ Critical Load Error: {e}")

# Classes List (52 Classes)
CLASSES = ['Apple Brown_spot', 'Apple Normal', 'Apple black_spot', 'Apricot Normal', 'Apricot blight leaf disease', 'Apricot shot_hole', 'Bean Fungal_leaf disease', 'Bean Normal leaf', 'Bean bean rust image', 'Bean shot_hole', 'Cherry Leaf Scorch', 'Cherry Normal leaf', 'Cherry brown_spot', 'Cherry purple leaf spot', 'Cherry_shot hole disease', 'Corn Fungal leaf', 'Corn Normal leaf', 'Corn gray leaf spot', 'Corn holcus_ leaf spot', 'Fig Blight_leaf disease', 'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf', 'Grape Anthracnose leaf', 'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease', 'Lokat Normal leaf', 'Pear Black spot _ leaf disease', 'Pear Normal _leaf', 'Pear fire blight', 'Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 'Walnut Normal_leaf', 'Walnut Shot_hole', 'Walnut leaf gall mite', 'lokat Leaf_spot', 'persimmons Brown_spot', 'tomato Fusarium Wilt', 'tomato spider mites', 'tomato verticillium wilt', 'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy_leaf', 'tomato_late_blight', 'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf']

def get_ai_analysis(img_bytes, cnn_result):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        prompt = f"""
        Analyze this plant leaf image. The CNN suggests: {cnn_result}.
        1. If NOT a leaf, reply 'INVALID'.
        2. If IS a leaf, explain Cause and Cure for a Pakistani farmer.
        Keep it simple with Urdu summary.
        """
        response = ai_engine.generate_content([prompt, img])
        return response.text
    except: return "Analysis unavailable."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    try:
        file = request.files['file']
        img_bytes = file.read()
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        img_array = preprocess_input(np.array(img).astype('float32'))
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            preds = model.predict(img_array)
            confidence = np.max(preds[0])
            index = np.argmax(preds[0])
            cnn_disease = CLASSES[index]

            ai_details = get_ai_analysis(img_bytes, cnn_disease)

            if "INVALID" in ai_details.upper() and confidence < 0.45:
                return jsonify({"disease": "Object Not Recognized", "status": "invalid", "urdu": "اے آئی اسے پودا تسلیم نہیں کر رہی۔"})

            return jsonify({
                "disease": cnn_disease,
                "confidence": f"{round(confidence * 100, 1)}%",
                "details": ai_details,
                "status": "healthy" if "Normal" in cnn_disease or "healthy" in cnn_disease.lower() else "danger"
            })
        return jsonify({"error": "Model offline"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
