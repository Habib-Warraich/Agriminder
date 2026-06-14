import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Sirf errors dikhao, warnings nahi
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Keras specific imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

# --- 1. MODEL LOADING (STABLE) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

model = None
try:
    if os.path.exists(model_path):
        # compile=False is the only way to avoid BatchNormalization errors on Render
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ CNN Brain Loaded Successfully")
    else:
        print(f"⚠️ model.h5 not found at {model_path}")
except Exception as e:
    print(f"❌ Load Error: {e}")

# --- 2. 52 CLASSES LIST ---
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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    try:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img).astype('float32')
        # PRECISION: MobileNetV2 requires [-1, 1] scaling
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            predictions = model.predict(img_array)
            confidence = np.max(predictions[0])
            index = np.argmax(predictions[0])

            # LAPTOP GUARD: 75% Confidence Threshold
            if confidence < 0.75:
                return jsonify({
                    "disease": "Object Not Recognized",
                    "confidence": f"{round(confidence * 100, 1)}%",
                    "treatment": "Please scan a clear plant leaf. AI cannot verify this object.",
                    "urdu": "اے آئی اس چیز کو پودا تسلیم نہیں کر رہی۔ براہ کرم پتے کی صاف تصویر لیں۔"
                })

            result_name = CLASSES[index]
            is_healthy = "Normal" in result_name or "healthy" in result_name

            return jsonify({
                "disease": result_name,
                "confidence": f"{round(confidence * 100, 1)}%",
                "treatment": "Plant is healthy." if is_healthy else "Disease detected. Apply fungicide.",
                "urdu": "پودا صحت مند ہے۔" if is_healthy else f"تشخیص: {result_name}"
            })
        
        return jsonify({"error": "Model not loaded"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-govt-rates', methods=['GET'])
def get_rates():
    # Aapka Mandi rates wala code yahan ayega
    return jsonify([{"crop": "Wheat", "price": "3950"}])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
