import os
import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# --- 1. CNN MODEL SETUP ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.h5')

try:
    # compile=False avoids BatchNormalization version mismatch errors
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ CNN Model Loaded Successfully")
except Exception as e:
    model = None
    print(f"❌ Model Error: {e}")

CLASSES = ['Blight (جھلساؤ)', 'Healthy (صحت مند)', 'Leaf Spot (دھبے)', 'Yellow Rust (پیلا کنگی)']

# --- 2. GOVERNMENT RATE SCRAPER ---
def scrape_punjab_rates():
    url = "https://agripunjab.gov.pk/pricelist"
    try:
        # User-Agent header helps bypass bot detection
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for the rates table (usually the first table on this specific site)
            table = soup.find('table')
            data = []
            
            if table:
                rows = table.find_all('tr')
                for row in rows[1:]: # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        crop_name = cols[0].text.strip()
                        price = cols[1].text.strip()
                        data.append({
                            "crop": crop_name,
                            "price": price,
                            "unit": "40kg",
                            "status": "Official Rate"
                        })
            return data
        return []
    except Exception as e:
        print(f"Scraper Error: {e}")
        return []

# --- 3. ROUTES ---

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32')
        
        # Precise Normalization for MobileNetV2 standards
        img_array = (img_array / 127.5) - 1.0 
        img_array = np.expand_dims(img_array, axis=0)

        if model:
            preds = model.predict(img_array)
            conf = np.max(preds[0])
            idx = np.argmax(preds[0])

            # Validation Layer (Thresholding)
            if conf < 0.75:
                return jsonify({
                    "disease": "Object Not Recognized",
                    "confidence": "Low",
                    "treatment": "Please scan a clear plant leaf. AI could not verify this object.",
                    "urdu": "اے آئی اس چیز کو نہیں پہچان سکی۔ براہ کرم پتے کی صاف تصویر لیں۔"
                })

            return jsonify({
                "disease": CLASSES[idx],
                "confidence": f"{round(conf * 100, 2)}%",
                "treatment": "Apply Propiconazole fungicide. Improve field drainage.",
                "urdu": "تشخیص مکمل۔ متاثرہ حصے پر فنگسائڈ کا سپرے کریں۔"
            })
        return jsonify({"error": "Model not loaded"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-govt-rates', methods=['GET'])
def get_rates():
    data = scrape_punjab_rates()
    if not data:
        # Fallback for demo if scraper is blocked or internet is down
        data = [
            {"crop": "Wheat (Gandum)", "price": "3,900", "unit": "40kg", "status": "Govt Fixed"},
            {"crop": "Rice (Basmati)", "price": "9,200", "unit": "40kg", "status": "Mandi Rate"},
            {"crop": "Maize (Makai)", "price": "2,850", "unit": "40kg", "status": "Govt Fixed"}
        ]
    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)