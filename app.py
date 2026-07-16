import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# TFLite Runtime Import
try:
    import tflite_runtime.interpreter as tflite
except (ImportError, ModuleNotFoundError):
    import tensorflow.lite as tflite

app = Flask(__name__)
CORS(app)

# --- 1. GEMINI AI CONFIG ---
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
ai_engine = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. SMART TFLITE LOADING (Fixed Path) ---
# Ye logic file ko dhoonday ga chahay Render par path jo bhi ho
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.tflite')

interpreter = None
input_details = None
output_details = None

if os.path.exists(model_path):
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ CNN Brain Loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
else:
    # Agar model na milay toh terminal mein list dikhao taake debug ho sakay
    print(f"❌ CRITICAL ERROR: model.tflite NOT FOUND at {model_path}")
    print(f"Available files in current dir: {os.listdir(base_dir)}")

# ... (Keep your CLASSES list and Predict route exactly the same)

@app.route('/')
def home():
    if interpreter:
        return "AgriMinder AI: Online & Model Loaded! ✅"
    else:
        return "AgriMinder AI: Online but Model Missing! ❌"

# (Baaki predict route same rahay ga)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
