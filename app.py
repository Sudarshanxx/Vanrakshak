from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from detect_poaching import run_detection  # your function

app = Flask(__name__)
# Enable CORS so your frontend can call this API
CORS(app)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json(force=True)
        # Extract base64 payload
        img_b64 = data.get('image', '')
        header, encoded = img_b64.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Run your detection pipeline: returns (status, message, detections)
        # where detections = list of (label, confidence, (xmin,ymin,xmax,ymax))
        status, message, detections = run_detection(img)

        # Draw boxes on image
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for label, conf, box in detections:
            xmin, ymin, xmax, ymax = box
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
            text = f"{label} {conf:.2f}"
            text_size = font.getsize(text)
            draw.rectangle([xmin, ymin - text_size[1], xmin + text_size[0], ymin], fill='red')
            draw.text((xmin, ymin - text_size[1]), text, fill='white', font=font)

        # Encode annotated image back to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        annotated_b64 = base64.b64encode(buffered.getvalue()).decode()
        annotated_data = f"data:image/jpeg;base64,{annotated_b64}"

        return jsonify({'status': status, 'message': message, 'annotated_image': annotated_data}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)
