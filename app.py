from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import datetime
import threading
from detect_poaching import model, play_alert_sound, send_email, confidence_threshold, target_labels, weapon_labels

app = Flask(__name__)
CORS(app)

# Global video capture object and detection info
cap = cv2.VideoCapture(0)
last_detection = {
    'timestamp': '',
    'labels': []
}

def generate_frames():
    global last_detection
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        # Run detection every frame
        results = model(frame)
        df = results.pandas().xyxy[0]
        labels = df['name'].tolist()
        confidences = df['confidence'].tolist()
        boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values

        detected_labels = []
        for label, conf in zip(labels, confidences):
            if label in target_labels and conf > confidence_threshold:
                detected_labels.append(label)

        # If anything important is detected
        if detected_labels:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            filename = f'detection_{timestamp.replace(" ", "_").replace(":", "-")}.jpg'
            cv2.imwrite(filename, frame)

            # Update last detection info
            last_detection['timestamp'] = timestamp
            last_detection['labels'] = list(set(detected_labels))

            play_alert_sound()

            if 'person' in detected_labels:
                send_email("🚨 Alert: Person Detected", f"Person detected at {timestamp}", filename)
            if any(label in detected_labels for label in weapon_labels):
                send_email("🚨 Alert: Weapon Detected", f"Weapon detected at {timestamp}", filename)
            if 'person' in detected_labels and any(label in detected_labels for label in weapon_labels):
                send_email("🚨 Alert: Person with Weapon Detected", f"Person with weapon detected at {timestamp}", filename)

        # Draw boxes and labels
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            conf = confidences[i]
            label = labels[i]
            if conf > confidence_threshold:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # make sure you place index.html in templates/

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_detection')
def latest_detection_api():
    return jsonify({
        'timestamp': last_detection['timestamp'],
        'labels': last_detection['labels'],
        'detected': bool(last_detection['labels'])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)
