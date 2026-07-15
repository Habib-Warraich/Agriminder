import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# --- 1. GEMINI AI CONFIG ---
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
vision_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. TFLITE MODEL LOADING (Bypasses Keras Errors) ---
# Hum model.h5 ki bajaye disease_model.tflite use karenge jo aapke folder mein hai
base_dir = os.path.dirname(os.path.abspath(__file__))
tflite_path = os.path.join(base_dir, 'disease_model.tflite')

interpreter = None
input_details = None
output_details = None

try:
    if os.path.exists(tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ CNN Brain (TFLite) Loaded Successfully")
    else:
        print(f"❌ Error: {tflite_path} not found")
except Exception as e:
    print(f"❌ TFLite Load Error: {e}")

# Labels (Ensure this matches your 52 classes order)
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

def get_ai_analysis(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        prompt = "Analyze this image. If it is a plant leaf, identify it and any disease. Tell Cause and Cure in detail for a Pakistani farmer in Gujrat. If it is NOT a plant leaf, reply ONLY with 'INVALID'."
        response = vision_model.generate_content([prompt, img])
        return response.text
    except: return "AI analysis error. Please check your internet."

@app.route('/')
def home():
    return "AgriMinder AI Cloud: ONLINE"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    img_bytes = file.read()
    
    # 1. Universal AI Check
    ai_detail = get_ai_analysis(img_bytes)
    if "INVALID" in ai_detail:
        return jsonify({"disease": "Object Not Recognized", "status": "invalid", "urdu": "اے آئی اسے پودا تسلیم نہیں کر رہی۔"})

    # 2. CNN TFLite Inference
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        # MobileNetV2 Normalization (-1 to 1)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        if interpreter:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])
            
            confidence_val = np.max(preds[0])
            index = np.argmax(preds[0])
            disease = CLASSES[index]

            return jsonify({
                "disease": disease,
                "confidence": f"{round(confidence_val * 100, 1)}%",
                "details": ai_detail,
                "status": "healthy" if "Normal" in disease or "healthy" in disease else "danger"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Prediction Logic Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
