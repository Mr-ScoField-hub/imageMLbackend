from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import torch
import uuid
import werkzeug.utils

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # max 5MB uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def categorize_food(caption):
    c = caption.lower()
    if any(w in c for w in ['apple', 'banana', 'orange', 'grape', 'fruit']):
        return 'Fruit'
    if any(w in c for w in ['lettuce', 'carrot', 'broccoli', 'vegetable']):
        return 'Vegetable'
    if any(w in c for w in ['soda', 'coke', 'juice', 'bottle', 'drink']):
        return 'Beverage'
    if any(w in c for w in ['chips', 'snack', 'cookie', 'biscuit']):
        return 'Snack'
    return 'Unknown'

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = werkzeug.utils.secure_filename(image.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    image.save(filepath)

    try:
        raw_image = Image.open(filepath).convert('RGB')
        inputs = processor(images=raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        category = categorize_food(caption)

        return jsonify({
            'image_url': f'/static/uploads/{unique_filename}',
            'caption': caption,
            'category': category
        })
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
