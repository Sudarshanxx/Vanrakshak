import os
import urllib.request
import cv2
import torch
import pygame
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

# -----------------------------
# DOWNLOAD ALERT SOUND IF MISSING
# -----------------------------
if not os.path.exists("alert.mp3"):
    print("Downloading alert sound...")
    url = "https://www.soundjay.com/button/sounds/beep-07.mp3"
    urllib.request.urlretrieve(url, "alert.mp3")
    print("Download complete.")

# -----------------------------
# INITIALIZE PYGAME FOR SOUND
# -----------------------------
def play_alert_sound():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        print("Failed to play sound:", e)

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# -----------------------------
# LABELS
# -----------------------------
target_labels = [
    'person', 
    'deer', 'elephant', 'tiger', 'leopard', 'lion', 'bear', 'boar', 'monkey', 'wolf',
    'fox', 'panther', 'bison', 'rhinoceros', 'giraffe', 'crocodile', 'zebra', 'cheetah',
    'hyena', 'peacock',
    'car', 'bus', 'truck', 'motorbike', 'bicycle',
    'knife', 'rifle', 'pistol', 'shotgun', 'bow', 'crossbow', 'spear', 'machete', 'axe'
]
vehicle_labels = ['car', 'bus', 'truck', 'motorbike', 'bicycle']
weapon_labels = ['knife', 'rifle', 'pistol', 'shotgun', 'bow', 'crossbow', 'spear', 'machete', 'axe']

# -----------------------------
# EMAIL SETUP
# -----------------------------
EMAIL_SENDER = "sudarshantayde04@gmail.com"
EMAIL_PASSWORD = "gcbj kiaa kvhc qbvi"
EMAIL_RECEIVER = "piyushshirke91@gmail.com"

def send_email(subject, body, image_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with open(image_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={image_path}')
            msg.attach(part)

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# -----------------------------
# RESTRICTED AREA
# -----------------------------
restricted_area = [(100, 100), (500, 500)]

def is_in_restricted_area(xmin, ymin, xmax, ymax, area):
    x1, y1 = area[0]
    x2, y2 = area[1]
    return (xmin > x1 and ymin > y1 and xmax < x2 and ymax < y2)

# -----------------------------
# MAIN LOOP - WRAPPED
# -----------------------------
if __name__ == "__main__":
    # Initialize log
    with open('log.txt', 'a', encoding='utf-8') as log:
        log.write("Detection log started\n")

    cap = cv2.VideoCapture(0)
    frame_count = 0
    confidence_threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        if frame_count % 5 == 0:
            results = model(frame)
            df = results.pandas().xyxy[0]
            labels = df['name'].tolist()
            confidences = df['confidence'].tolist()
            boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values

            detected = []
            for label, confidence in zip(labels, confidences):
                if label in target_labels and confidence > confidence_threshold:
                    detected.append(label)

            if detected:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f'detection_{timestamp}.jpg'
                cv2.imwrite(filename, frame)

                log_entry = f"🚨 Detected: {detected} at {timestamp}\n"
                print(log_entry)
                with open('log.txt', 'a', encoding='utf-8') as log:
                    log.write(log_entry)

                play_alert_sound()

                if 'person' in detected:
                    send_email("🚨 Forest Alert: Person Detected", f"Person detected in forest area at {timestamp}!", filename)

                if any(item in detected for item in weapon_labels):
                    send_email("🚨 Forest Alert: Weapon Detected", f"Weapon detected at {timestamp}!", filename)

                for animal in detected:
                    if animal in target_labels and animal not in vehicle_labels + weapon_labels + ['person']:
                        send_email(f"🚨 Wild Animal Alert: {animal}", f"Wild Animal Detected: {animal} at {timestamp}!", filename)

                for label, conf, box in zip(labels, confidences, boxes):
                    if label in vehicle_labels and conf > confidence_threshold:
                        xmin, ymin, xmax, ymax = box
                        if is_in_restricted_area(xmin, ymin, xmax, ymax, restricted_area):
                            send_email(f"🚨 Restricted Area Alert: {label}", f"Vehicle Detected in Restricted Area: {label} at {timestamp}!", filename)
            else:
                print("No target object detected.")
                with open("log.txt", "a") as log:
                    log.write("No target object detected.\n")

            for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
                conf = confidences[i]
                label = labels[i]
                if conf > confidence_threshold:
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Surveillance Feed", frame)
        if cv2.waitKey(30) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
