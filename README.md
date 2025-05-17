<h1 align="center" style="font-size: 64px;">🧑‍💻 Face Recognition System</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen?logo=github&logoColor=white" alt="Contributions Welcome" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?logo=license&logoColor=white" alt="License" />
</div>

## 📖 About the Project

This project is an advanced face recognition system designed to capture, train, and recognize faces in real time. It uses Haar cascades for face detection and the FisherFace algorithm for face recognition. The system supports creating a face dataset by capturing multiple images, training a model, and then identifying known or unknown individuals via webcam input.

---

## 🚀 Features

- Face Detection: Detects faces using Haar cascades.  
- Dataset Creation: Captures 50 grayscale face images and saves them for training.  
- Face Recognition: Identifies individuals using the trained FisherFace recognizer.  
- Unknown Detection: Flags unknown faces and saves frames for review.

---

## 🛠 Prerequisites

- Python 3.x  
- OpenCV  
- NumPy

---

## ⚙️ How It Works

### 1. 🖼️ Face Dataset Creation

- Run the dataset creation script:
   
  ```bash
  python Capture Face Images.py
  
The script opens the webcam, detects faces, and saves 50 grayscale images to the datasets/New Data directory.

### 2. 🤖 Face Recognition

Run the face recognition script:

```bash
python Realtime Face Recognition.py
```
The script loads the Haar cascade and FisherFace recognizer, detects faces in real time, and identifies known or unknown faces.

---

## 🖥 Scripts Overview

### 📸 face_dataset_creator.py

- Captures and saves face images for training.  
- Creates dataset directory if not present.  
- Resizes face images to 130x100 pixels.

### 🤖 face_recognition.py

- Loads the trained FisherFace recognizer.  
- Identifies known faces or flags unknown faces.  
- Saves frames of unknown faces for analysis.

---

## 🛡 Dependencies

- Install required packages:  
  ```bash
  pip install opencv-python numpy
---
## 🎯 Use Case

- Secure access control to restricted areas.  
- Attendance management in schools and workplaces.  
- Enhancing user authentication for applications.  
- Monitoring unknown persons in surveillance systems.

---

## 🤝 Contribution

- Report issues or bugs.  
- Suggest new features or improvements.  
- Submit pull requests with enhancements or fixes.  

Please follow best coding practices and include documentation with contributions.

---

## 📜 License

Licensed under the MIT License. See the License file for the details
