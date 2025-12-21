# OCR University Project

This repository contains two distinct Optical Character Recognition (OCR) projects developed for a university curriculum. It showcases a transition from high-level API integration to low-level model training using Deep Learning.

## 📝 Project Description
This repository features two OCR projects: **V1** is a Python-based interface integrating Tesseract for local text extraction and the Gemini API for advanced refinement. **V2** focuses on deep learning, featuring a custom CNN model trained on the A-Z dataset to recognize handwritten characters. A complete university-level computer vision portfolio.

---

## 📂 Repository Structure

### **V1: Hybrid OCR & Gemini Refinement**
A practical application designed to handle real-world documents where standard OCR often produces "noisy" results.
* **Core Engine:** `Pytesseract` (Tesseract OCR) for baseline text extraction.
* **AI Enhancement:** Integrated with the **Google Gemini API** to post-process text, correcting spelling mistakes and improving semantic formatting.
* **Interface:** A Python-based GUI/Web interface for easy image uploading and result comparison.

### **V2: Custom A-Z Character Recognition**
A deep learning project focused on building an OCR engine from the ground up.
* **Model Architecture:** A Convolutional Neural Network (CNN) built with `TensorFlow` and `Keras`.
* **Dataset:** Trained on the **A-Z Handwritten Alphabets** dataset.
* **Pipeline:** Includes image preprocessing (grayscaling, resizing to $28 \times 28$, and normalization) and model evaluation metrics.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* **Python 3.10+**
* **Tesseract OCR Engine:** [Install Tesseract](https://github.com/tesseract-ocr/tesseract) and ensure it is added to your System PATH.
* **Gemini API Key:** Obtain a key from the [Google AI Studio](https://aistudio.google.com/).

### 2. Clone the Repository
```bash
git clone [https://github.com/aoulichak/OCR-University-Project.git](https://github.com/aoulichak/OCR-University-Project.git)
cd OCR-University-Project
```
### 3. Install Dependencies
**Bash**
```bash
pip install pytesseract google-generativeai tensorflow opencv-python pillow matplotlib
```
### 🚀 How to Run

#### **Running Project V1**
1. Navigate to the `V1` folder.
2. Add your **Gemini API key** to the configuration file or environment variables.
3. Run the application:
```bash
python main.py
```
#### **Running Project V2**
1. Ensure the **A-Z dataset** (CSV or images) is placed in the `data/` directory.
2. Train the model:
```bash
python train.py
```
Test predictions:
```bash
python predict.py --image path/to/char.png
```
### Technical Summary

| Feature | Project V1 | Project V2 |
| :--- | :--- | :--- |
| **Method** | Pre-trained + LLM | Custom CNN Training |
| **Input** | Full Document Images | Single Handwritten Characters |
| **Key Tech** | Tesseract, Gemini API | TensorFlow, Keras, OpenCV |
| **Complexity** | High-level Integration | Low-level Deep Learning |
