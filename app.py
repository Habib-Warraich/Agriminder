import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# TFLite Runtime Import Fix
try:
    import tflite_runtime.interpreter as tflite
except (ImportError, ModuleNotFoundError):
    # Agar laptop par install na ho toh fallback to tensorflow
    import tensorflow.lite as tflite

app = Flask(__name__)
CORS(app)

# --- 1. GEMINI AI CONFIG ---
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
ai_engine = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. TFLITE MODEL LOADING (Absolute Path) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.tflite')

interpreter = None
input_details = None
output_details = None

try:
    if os.path.exists(model_path):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ TFLite Model Loaded Successfully")
    else:
        print(f"❌ Error: {model_path} not found")
except Exception as e:
    print(f"❌ Interpreter Error: {e}")

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

def get_ai_analysis(img_bytes, cnn_result):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        prompt = f"Analyze this leaf. Model says: {cnn_result}. If NOT a plant, reply 'INVALID'. Otherwise tell Cause and Cure for a Pakistani farmer in Gujrat. Add Urdu summary."
        response = ai_engine.generate_content([prompt, img])
        return response.text
    except:
        return "AI analysis busy. Consult NPK guide."

@app.route('/')
def home():
    return "AgriMinder TFLite Backend is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    try:
        file = request.files['file']
        img_bytes = file.read()
        
        # Pre-processing
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        img_array = np.array(img).astype('float32')
        img_array = (img_array / 127.5) - 1.0 # Normalization
        img_array = np.expand_dims(img_array, axis=0)

        if interpreter:
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            confidence = np.max(predictions[0])
            idx = np.argmax(predictions[0])
            cnn_disease = CLASSES[idx]

            ai_details = get_ai_analysis(img_bytes, cnn_disease)

            if confidence < 0.60 and "INVALID" in ai_details.upper():
                return jsonify({"disease": "Object Not Recognized", "status": "invalid", "urdu": "پودا نہیں ملا۔"})

            return jsonify({
                "disease": cnn_disease,
                "confidence": f"{round(confidence * 100, 1)}%",
                "details": ai_details,
                "status": "healthy" if "Normal" in cnn_disease else "danger"
            })
        
        return jsonify({"error": "Model offline"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render dynamic port support
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
