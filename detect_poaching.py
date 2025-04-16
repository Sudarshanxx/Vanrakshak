import os
import urllib.request
import torch
import pygame
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ----------------------------- ALERT SOUND -----------------------------
if not os.path.exists("alert.mp3"):
    print("Downloading alert sound...")
    url = "https://www.soundjay.com/button/sounds/beep-07.mp3"
    urllib.request.urlretrieve(url, "alert.mp3")
    print("Download complete.")

def play_alert_sound():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        print("Failed to play sound:", e)

# ----------------------------- YOLO MODEL -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# ----------------------------- LABELS -----------------------------
target_labels = [
    'person', 'deer', 'elephant', 'tiger', 'leopard', 'lion', 'bear', 'boar', 'monkey',
    'wolf', 'fox', 'panther', 'bison', 'rhinoceros', 'giraffe', 'crocodile', 'zebra',
    'cheetah', 'hyena', 'peacock', 'car', 'bus', 'truck', 'motorbike', 'bicycle',
    'knife', 'rifle', 'pistol', 'shotgun', 'bow', 'crossbow', 'spear', 'machete', 'axe'
]
vehicle_labels = ['car', 'bus', 'truck', 'motorbike', 'bicycle']
weapon_labels = ['knife', 'rifle', 'pistol', 'shotgun', 'bow', 'crossbow', 'spear', 'machete', 'axe']
confidence_threshold = 0.5

# ----------------------------- EMAIL -----------------------------
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
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(part)

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)

# ----------------------------- RUN DETECTION -----------------------------
def run_detection(pil_image):
    try:
        results = model(pil_image)
        df = results.pandas().xyxy[0]
        detections = []

        for _, row in df.iterrows():
            label = row['name']
            conf = row['confidence']
            if conf > confidence_threshold and label in target_labels:
                bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                detections.append((label, conf, bbox))

        if detections:
            return 'success', 'Objects detected', detections
        else:
            return 'success', 'No target objects detected', []

    except Exception as e:
        return 'error', str(e), []
