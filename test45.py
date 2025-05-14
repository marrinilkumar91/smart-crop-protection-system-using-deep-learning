import cv2
import numpy as np
import os
import pygame
from threading import Thread
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time
from twilio.rest import Client

# Initialize pygame mixer
pygame.mixer.init()

# Global variables to track state
sound_playing = False
animal_detected = False
person_detected = False
start_time_animal = None
start_time_person = None
detected_animal_name = None  # Variable to store the name of the detected animal

# Email configuration
email_user = 'agriculture19092002@gmail.com'
email_password = 'cppu iywx lagm kstk'
email_send = '21p61a6638@vbithyd.ac.in'

# Twilio configuration
account_sid = 'ACe565aa37b42ba99c3666f9eb2db25a23'
auth_token = '3c64a0ab1994ce1d730c16efc618908e'
twilio_number = '+18286726429'
farmer_number = '+918008576756'

# Animal detection model configuration
cap = cv2.VideoCapture(0)
wht = 640  # Increased frame width and height for better interface
hgt = 480
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = r'C:\Users\Anil kumar Marri\OneDrive\Desktop\mini project\coco.names'
modelConfi = r'C:\Users\Anil kumar Marri\OneDrive\Desktop\mini project\yolov3.cfg'
modelwei = r'C:\Users\Anil kumar Marri\OneDrive\Desktop\mini project\yolov3.weights'

classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet(modelConfi, modelwei)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Face recognition model configuration
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
registered_face_image_path = r"C:\Users\Anil kumar Marri\OneDrive\Pictures\Camera Roll\WIN_20240911_12_35_55_Pro.jpg"
registered_face_img = cv2.imread(registered_face_image_path, cv2.IMREAD_GRAYSCALE)

registered_face_cascade = face_cascade.detectMultiScale(registered_face_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
if len(registered_face_cascade) == 0:
    raise ValueError("No face detected in the registered image.")

x, y, w, h = registered_face_cascade[0]
registered_face = registered_face_img[y:y+h, x:x+w]

# ORB detector for face recognition
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(registered_face, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Function to play sound based on animal type
def play_sound(sound_path):
    global sound_playing
    sound_playing = True
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
    except Exception as e:
        print(f"Failed to play sound: {e}")
    sound_playing = False

# Function to send email
def send_email(subject, body, attachment_path=None):
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_send
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    if attachment_path:
        # Open the file in binary mode
        with open(attachment_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)  # Encode the payload
            part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(attachment_path)}')
            msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_user, email_password)
        text = msg.as_string()
        server.sendmail(email_user, email_send, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to make phone call
def make_phone_call():
    client = Client(account_sid, auth_token)
    try:
        call = client.calls.create(
            twiml='<Response><Say>Alert: An issue has been detected on your property. Please check immediately.</Say></Response>',
            from_=twilio_number,
            to=farmer_number
        )
        print("Call initiated successfully")
    except Exception as e:
        print(f"Failed to make phone call: {e}")

# Function to handle object detection
def findobj(outputs, img):
    global animal_detected, start_time_animal, detected_animal_name

    ht, wt, ct = img.shape
    bbox = []
    classIds = []
    confs = []
    detected = False

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wt), int(det[3] * ht)
                x, y = int((det[0] * wt) - w / 2), int((det[1] * ht) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                detected = True

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 2))
            detected_animal_name = classNames[classIds[i]]  # Store the detected animal's name

            if detected_animal_name in ["bird", "horse", "cow", "elephant", "bear", "zebra", "giraffe"]:
                if not sound_playing:
                    Thread(target=play_sound, args=(r'C:\Users\Anil kumar Marri\OneDrive\Desktop\mini project\firecrackers-178799.mp3',)).start()
            elif detected_animal_name in ["cat", "dog", "sheep"]:
                if not sound_playing:
                    Thread(target=play_sound, args=(r'C:\Users\Anil kumar Marri\OneDrive\Desktop\mini project\lion-roars-with-growls-and-inhales-195839.mp3',)).start()

    if detected:
        if not animal_detected:
            animal_detected = True
            start_time_animal = time.time()
        elif time.time() - start_time_animal > 60:
            email_subject = f'Animal Detected for 1 Minute: {detected_animal_name.upper()}'
            email_body = f'An animal ({detected_animal_name.upper()}) has been detected continuously for 1 minute. Please check your property.'
            # Capture image before sending email
            captured_image_path = capture_image(img)
            send_email(email_subject, email_body, captured_image_path)  # Send the email with the image
            make_phone_call()
            start_time_animal = time.time()
    else:
        animal_detected = False
        start_time_animal = None

# Function to capture image
def capture_image(frame):
    captured_image_path = 'captured_image.jpg'
    cv2.imwrite(captured_image_path, frame)  # Save the captured image
    return captured_image_path

# Function to handle face recognition
def recognize_person(frame):
    global person_detected, start_time_person

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        keypoints2, descriptors2 = orb.detectAndCompute(face, None)

        if descriptors2 is not None and len(descriptors2) > 0:
            matches = bf.match(descriptors1, descriptors2)
            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) > 10:  # Check if enough good matches are found
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Owner Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
                person_detected = True
                start_time_person = time.time()
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
                person_detected = False
                start_time_person = None
        else:
            person_detected = False
            start_time_person = None

    if person_detected:
        if time.time() - start_time_person > 60:
            # No action needed as the owner is detected
            pass
    else:
        # Check if unknown person detected for 1 minute
        if start_time_person and time.time() - start_time_person > 60:
            email_subject = 'Unknown Person Detected'
            email_body = 'An unknown person has been detected on your property for 1 minute. Please check immediately.'
            # Capture image before sending email
            captured_image_path = capture_image(frame)
            send_email(email_subject, email_body, captured_image_path)  # Send the email with the image
            make_phone_call()
            start_time_person = time.time()

# Function to process frames from the camera
def main():
    global start_time_person, start_time_animal

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (wht, hgt))  # Resize frame for better performance

        # Object detection
        blob = cv2.dnn.blobFromImage(img, 0.00392, (wht, hgt), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        # Find objects in the frame
        findobj(outputs, img)

        # Recognize person in the frame
        recognize_person(img)

        # Show the frame
        cv2.imshow("Video Feed", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
