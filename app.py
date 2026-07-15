import os
import io
import time
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# --- 1. THE ONLY PROPER FIX FOR 'InputLayer' ERROR ---
# Yeh code Keras 3 ko bataye ga ke 'batch_shape' aur 'optional' ko ignore karay
from tensorflow.keras.layers import InputLayer, BatchNormalization

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

# Original class ko replace kar do loading se pehle
tf.keras.layers.InputLayer = PatchedInputLayer
# -----------------------------------------------------

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# Gemini AI Key
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
ai_engine = genai.GenerativeModel('gemini-1.5-flash')

# Model Path Logic
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

model = None
try:
    if os.path.exists(model_path):
        # compile=False and trust the patch above
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ CNN Brain Loaded Successfully with FYP-Patch")
    else:
        print("❌ Error: model.h5 not found")
except Exception as e:
    print(f"❌ Load Error: {e}")

# Classes (Wahi 52 jo aapne train ki hain)
CLASSES = ['Apple Brown_spot', 'Apple Normal', 'Apple black_spot', 'Apricot Normal', 
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
    'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf']

def get_ai_advice(name):
    try:
        p = f"Analyze plant disease: {name}. Tell Cause and Cure in 3 lines for a farmer in Gujrat, Pakistan. Add 1 line Urdu."
        return ai_engine.generate_content(p).text
    except: return "Consult expert."

@app.route('/')
def index(): return "Server Live"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32')
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            predictions = model.predict(img_array)
            confidence = np.max(predictions[0])
            index = np.argmax(predictions[0])

            # Laptop Screen Guard (If confidence is low)
            if confidence < 0.75:
                return jsonify({"disease": "Object Not Recognized", "status": "invalid", "urdu": "پودا نہیں ملا۔"})

            disease = CLASSES[index]
            details = get_ai_advice(disease)
            return jsonify({
                "disease": disease,
                "confidence": f"{round(confidence * 100, 1)}%",
                "details": details,
                "status": "healthy" if "Normal" in disease else "danger"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
