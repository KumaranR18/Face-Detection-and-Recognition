import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'  # Haar cascade for face detection
datasets = 'datasets'  # Directory for datasets
sub_data = 'New Data'  # Subfolder name for the person

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path): os.mkdir(path)  # Create directory if it doesn't exist

(width, height) = (130, 100)  # Dimensions for resized face images
face_cascade = cv2.CascadeClassifier(haar_file)  # Load Haar cascade
webcam = cv2.VideoCapture(0)  # Initialize webcam

count = 1
while count <= 50:  # Capture 50 images
    _, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_resize = cv2.resize(gray[y:y + h, x:x + w], (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)  # Save resized face
        count += 1

    cv2.imshow('OpenCV', im)
    if cv2.waitKey(10) == 27: break  # Exit on pressing 'Esc'

webcam.release()
cv2.destroyAllWindows()
