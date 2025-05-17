import cv2
import numpy as np
import os

haar_file = 'haarcascade_frontalface_default.xml'  # Haar cascade for face detection
datasets = 'datasets'  # Path to the dataset directory

print('Training....')
(images, labels, names, id) = ([], [], {}, 0)  # Initialize variables

# Load images and labels from dataset
for (subdir, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir  # Map ID to name
        subjectpath = os.path.join(datasets, subdir)  # Path to individual folder
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)  # Full path of image
            label = id
            images.append(cv2.imread(path, 0))  # Load image in grayscale
            labels.append(int(label))
        id += 1

(width, height) = (130, 100)  # Dimensions for resized face
(images, labels) = [np.array(lis) for lis in [images, labels]]  # Convert to numpy arrays

# Initialize and train the FisherFace recognizer
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)  # Load Haar cascade
webcam = cv2.VideoCapture(0)  # Start webcam

cnt = 0  # Counter for unknown faces

while True:
    (_, im) = webcam.read()  # Read frame from webcam
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  # Extract face region
        face_resize = cv2.resize(face, (width, height))  # Resize to model's input size

        prediction = model.predict(face_resize)  # Predict using the trained model
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Draw rectangle around face

        if prediction[1] < 800:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), 
                        (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))  # Display name and confidence
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))  # Display "Unknown"
            if cnt > 100:  # Save frame if unknown detected frequently
                print("Unknown Person")
                cv2.imwrite("input.jpg", im)
                cnt = 0

    cv2.imshow('OpenCV', im)  # Display the frame
    key = cv2.waitKey(10)  # Exit on pressing 'Esc'
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
