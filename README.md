# AI-Based Biodegradable Waste Detection and Sorting System

This project presents an AI-powered solution for sorting biodegradable waste—especially rotten fruits—using Convolutional Neural Networks (CNNs). The system automates the detection and classification process through a conveyor system, sensors, and image processing to enhance waste management efficiency and accuracy.

---

## 🧠 Project Objective

To develop an intelligent system capable of:
- Detecting rotten (biodegradable) and fresh (non-biodegradable) fruits
- Sorting them via motorized arms on a conveyor system
- Reducing human effort and improving the accuracy of food waste classification

---

## ⚙️ System Components

- **IR Sensor** – Detects presence of fruit
- **Camera Module** – Captures image for analysis
- **CNN Model** – Classifies the fruit as fresh or rotten
- **Microcontroller (Arduino UNO)** – Coordinates the sorting mechanism
- **Conveyor & Motor Mechanism** – Physically sorts the fruits

---

## 🛠️ Technologies Used

- **Python 3.6.5**
- **OpenCV**
- **TensorFlow / Keras**
- **NumPy / Pandas / scikit-learn**
- **Arduino IDE**
- **Camera & IR Sensor Modules**

---

## 🖼️ How It Works

1. Fruit placed on conveyor
2. IR sensor detects object, halts conveyor
3. Camera captures fruit image
4. CNN model classifies image:
   - **Rotten → Compost bin**
   - **Fresh → Reuse bin**
5. Conveyor resumes

---

## 🧪 Dataset

A labeled dataset of **fresh and rotten fruit images** is used for:
- **Training**
- **Testing**
- **Validation**

Includes image preprocessing steps: scaling, normalization, and augmentation.

---

## 📦 Project Structure

├── dataset/
│ ├── fresh/
│ └── rotten/
├── model/
│ ├── cnn_model.h5
│ └── training_script.py
├── main.py
├── camera_module.py
├── arduino_interface/
├── README.md


---

## 🚀 How to Run

python main.py

### 🔧 Prerequisites

- Python 3.6+
- pip (Python package installer)
- Arduino IDE (for microcontroller programming)

### 📥 Installation

```bash
git clone https://github.com/yourusername/ai-waste-sorting.git
cd ai-waste-sorting
pip install -r requirements.txt
