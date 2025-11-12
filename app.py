from flask import Flask, render_template, request, send_from_directory, send_file, jsonify
import os
import shutil
import cv2
import numpy as np
from enhancer import enhance_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
TEMP_PREVIEWS = 'temp_previews'

# --- Limpiar carpeta temporal al iniciar ---
if os.path.exists(TEMP_PREVIEWS):
    shutil.rmtree(TEMP_PREVIEWS)
os.makedirs(TEMP_PREVIEWS, exist_ok=True)

# Crear carpetas de uploads y procesadas si no existen
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Perfiles predefinidos
PROFILES = {
    "Brasil": dict(brightness=20, contrast=1.2, sharpness=1.7, saturation=1.3, gamma=1.1),
    "Tokio": dict(brightness=10, contrast=1.5, sharpness=2.0, saturation=1.1, gamma=0.9),
    "Autumn": dict(brightness=5, contrast=1.1, sharpness=1.4, saturation=1.5, gamma=1.0),
    "Sunday": dict(brightness=15, contrast=1.3, sharpness=1.6, saturation=1.2, gamma=1.0),
    "Winday": dict(brightness=-5, contrast=1.0, sharpness=1.3, saturation=1.0, gamma=1.2)
}

def compute_suggested_params(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = int(np.clip(np.mean(gray) - 127, -100, 100))
    contrast = float(np.clip(np.std(gray) / 64, 0.5, 3.0))
    sharpness = 1.5
    saturation = 1.0
    mean_norm = np.mean(gray) / 255.0
    gamma = float(np.clip(np.log(0.5)/np.log(mean_norm), 0.1, 3.0)) if mean_norm > 0 else 1.0

    return dict(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        saturation=saturation,
        gamma=gamma,
        color_temp=0,
        edge_mark=0
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file or file.filename == '':
        return 'No file selected', 400

    filename = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(PROCESSED_FOLDER, filename)
    file.save(input_path)

    suggested = compute_suggested_params(input_path)
    enhance_image(input_path, output_path, **suggested)

    # Generar previews temporales para los perfiles
    previews = {"Manual": "/processed/" + filename}
    for prof, params in PROFILES.items():
        temp_path = os.path.join(TEMP_PREVIEWS, f"{prof}_{filename}")
        enhance_image(input_path, temp_path, **params)
        previews[prof] = f'/temp_previews/{prof}_{filename}'

    return jsonify({
        'original': f'/uploads/{filename}',
        'processed': f'/processed/{filename}',
        'filename': filename,
        'suggested': suggested,
        'previews': previews
    })

@app.route('/adjust', methods=['POST'])
def adjust():
    data = request.get_json()
    filename = data.get('filename')
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(PROCESSED_FOLDER, filename)

    enhance_image(
        input_path,
        output_path,
        brightness=data.get('brightness', 0),
        contrast=data.get('contrast', 1.0),
        sharpness=data.get('sharpness', 1.5),
        saturation=data.get('saturation', 1.0),
        gamma=data.get('gamma', 1.0),
        color_temp=data.get('color_temp', 0),
        edge_mark=data.get('edge_mark', 0)
    )
    return send_file(output_path, mimetype='image/jpeg')

@app.route('/apply_profile', methods=['POST'])
def apply_profile():
    data = request.get_json()
    profile = data.get('profile')
    filename = data.get('filename')
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(PROCESSED_FOLDER, filename)

    if profile == "Manual":
        # Modo manual: reaplica sliders actuales
        return send_file(output_path, mimetype='image/jpeg')

    if profile in PROFILES:
        enhance_image(input_path, output_path, **PROFILES[profile])
        return send_file(output_path, mimetype='image/jpeg')
    return 'Invalid profile', 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/temp_previews/<filename>')
def temp_preview(filename):
    return send_from_directory(TEMP_PREVIEWS, filename)

if __name__ == '__main__':
    app.run(debug=True)
