
# 🧠 AI Image Authenticity Detector

A machine learning project that detects whether an image is **real** or **AI-generated** using progressively built models — starting from raw pixel data to more advanced deep learning techniques.

---

## 🚀 Project Goal

With the rise of generative AI, distinguishing real images from AI-generated ones has become important.
This project aims to build a classifier that can automatically detect image authenticity.

---

## 🏗️ Project Roadmap

### 🟢 Level 1 — Dataset Handling & Preprocessing

* Load images from folders (`real/`, `fake/`)
* Resize images to uniform size (128x128)
* Convert images to RGB format
* Convert images into numpy arrays
* Create dataset:

  * `X` → image data
  * `y` → labels (0 = real, 1 = fake)

---

### 🟡 Level 2 — First Basic Classifier

* Flatten image arrays into 1D vectors
* Train a simple model:

  * Logistic Regression / Basic Neural Network
* Build full ML pipeline:

  ```
  Input → Model → Prediction
  ```
* Evaluate basic accuracy (not expected to be high)

---

### 🔵 Level 3 — Improving the Model

* Normalize pixel values
* Split dataset:

  * Training set
  * Testing set
* Try better models:

  * Deeper Neural Networks
* Tune parameters

---

### 🔴 Level 4 — CNN (Convolutional Neural Network)

* Move from flattened input → spatial understanding
* Build CNN using:

  * TensorFlow / PyTorch
* Learn:

  * Feature extraction
  * Convolutions
* Achieve significantly better accuracy

---

### 🟣 Level 5 — Real-World Improvements

* Data augmentation
* Handle overfitting
* Improve generalization
* Test on unseen/generated images

---

## 📂 Project Structure

```
AI-Generated-Image-Classifier/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── model.py
│
├── dataset/        # (Not included in repo)
│   ├── real/
│   └── fake/
│
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset is **not included** in this repository due to size limitations.

You can use:

* CIFAR-10 / ImageNet subset (for real images)
* AI-generated datasets from Kaggle

After downloading, structure it as:

```
dataset/
   real/
   fake/
```

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pillow (PIL)
* Scikit-learn
* TensorFlow / PyTorch (for advanced stages)

---

## 📌 Current Status

✔ Level 1: Completed
⏳ Level 2: In Progress
⬜ Level 3+: Planned

---

## 💡 Key Learnings

* Image preprocessing fundamentals
* How images are represented as numerical data
* Building an ML pipeline from scratch
* Transition from basic models → deep learning

---

## 🔮 Future Improvements

* Deploy as a web app
* Real-time image detection
* Use pre-trained models (ResNet, EfficientNet)
* Improve accuracy with larger datasets

---

## 👨‍💻 Author

**Nandan Nalwade**

---

## ⭐ Notes

This project is built as a **learning-first approach**, focusing on understanding each step deeply rather than using pre-built solutions.

---
