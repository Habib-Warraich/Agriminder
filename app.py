import os
import io
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# --- 1. CONFIGURATION ---
# Aapki API Key pehle se integrated hai
genai.configure(api_key="AQ.Ab8RN6JuGn3X0JCtey2b45h0KA3DHKFTUUskqLfo6fq5XU4EBA")
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/')
def home():
    return "AgriMinder AI Server is Online! Ready for Gujrat District Scan."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # --- UNIVERSAL AI VISION PROMPT ---
        # Yeh prompt patti ko pehchanay ga aur baki cheezon (laptop/person) ko reject karega
        prompt = """
        You are a high-precision Agricultural AI for Pakistan.
        Look at this image:
        1. If it is NOT a plant leaf, reply with: {"disease": "Object Not Recognized", "status": "invalid"}
        2. If it is a leaf, identify the plant and disease. 
        3. Explain the Cause (e.g. water issue, NPK deficiency).
        4. Give a detailed Cure (Pesticide names available in Pakistan).
        5. Provide a summary in Urdu.
        Return the result in JSON format only.
        """
        
        response = model.generate_content([prompt, img])
        
        # Cleanup response text to ensure it's valid JSON
        res_text = response.text.replace('```json', '').replace('```', '').strip()
        
        return res_text # Seedha JSON bhej raha hai mobile app ko

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "disease": "System Busy",
            "status": "danger",
            "details": "AI server is processing. Please try again.",
            "urdu": "سرور مصروف ہے۔ براہ کرم دوبارہ کوشش کریں۔"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
