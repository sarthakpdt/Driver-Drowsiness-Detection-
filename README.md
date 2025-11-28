# Driver Drowsiness Detection System 😴🚗  
A real-time system that detects driver drowsiness using **Eye CNN + Mouth CNN + Face Landmarks**, along with **confusion matrix**, **training loss graph**, and **alerts**.

---

## 📌 Features
- Real-time webcam detection  
- Eye CNN model for blink detection  
- Mouth CNN for yawning detection  
- Face landmarks to support eye/mouth region extraction  
- Confusion Matrix plotted after testing  
- Epoch vs Loss curve  
- Alarm system when driver looks drowsy  
- Works smoothly on CPU/GPU  

---

## 🧠 Project Structure
```
📂 Driver-Drowsiness-Detection
│── eye_cnn.py
│── mouth_cnn.py
│── train_eye_cnn.py
│── train_mouth_cnn.py
│── detector.py   (final main file)
│── utils.py
│── dataset/
│── models/
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```
pip install opencv-python mediapipe tensorflow keras numpy matplotlib scikit-learn
```

### 2️⃣ Train the Mouth Model
```
python train_mouth_cnn.py
```

### 3️⃣ Train the Eye Model
```
python train_eye_cnn.py
```

### 4️⃣ Run Real-Time Detection
```
python detector.py
```

---

## 📊 Output Graphs

### ✔ Confusion Matrix  
Automatically saved as:  
```
outputs/confusion_matrix.png
```

### ✔ Epoch vs Loss Curve  
Automatically saved as:  
```
outputs/loss_curve.png
```

---

## 🖼 Sample Training Graphs  
(Add screenshots here)

---

## 📁 Dataset
You can find the dataset link in the report organize the  dataset like this:
```
dataset/
│── mouth/
│     ├── yawn/
│     └── no_yawn/
│── eyes/
      ├── open/
      └── closed/
```

---

## 🛑 Drowsiness Logic
- If **eyes closed** for > 6 consecutive frames → Warning  
- If **mouth yawning** for > 10 frames → Drowsy alert  
- Final decision = Combined Eye + Mouth score  

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- Mediapipe  
- Scikit-Learn  
- Matplotlib  

---

## 📞 Contact  
For queries or collaboration:  
**Sarthak Pandit**

---

## ⭐ If you like this project, give it a star on GitHub!
