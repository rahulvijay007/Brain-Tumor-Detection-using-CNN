# Brain Tumor Detection using CNN

## Overview
This project aims to classify brain MRI images into four categories: **glioma, meningioma, pituitary tumor, and no tumor** using a **Convolutional Neural Network (CNN)**. The model is trained on a dataset of MRI images and is capable of automatically identifying the presence and type of brain tumors.

## Dataset
The dataset used for training and testing the model is publicly available on Kaggle:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Dataset Structure
```
Dataset
│── Testing
│   ├── glioma
│   ├── meningioma
│   ├── notumor
│   ├── pituitary
│── Training
│   ├── glioma
│   ├── meningioma
│   ├── notumor
│   ├── pituitary
```
Each subfolder contains MRI images corresponding to the respective class.

## Project Features
- **Data Preprocessing & Augmentation**: Image resizing, normalization, and augmentation techniques to improve model performance.
- **CNN Model Architecture**: A deep learning model built using TensorFlow/Keras.
- **Model Training & Evaluation**: The model is trained with proper callbacks for early stopping and best model checkpointing.
- **Performance Metrics**: Classification report, confusion matrix, and model accuracy/loss graphs.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow & Keras
- OpenCV
- NumPy, Pandas, Matplotlib, Seaborn, and Sci-kit Learn

### Install Required Packages
You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Run the Model
```bash
python brain_tumor_cnn.py
```

## Results
The model is evaluated using classification metrics and visualized through accuracy/loss graphs and a confusion matrix.

## Model Evaluation
- **Classification Report**
- **Confusion Matrix**
- **ROC Curve & AUC Score**

## Future Improvements
- Enhance data augmentation techniques.
- Use advanced CNN architectures like VGG16, ResNet, or EfficientNet.
- Implement a web-based interface for real-time predictions.

## License
This project is open-source and available for educational and research purposes.

---
📌 **Note**: Please ensure the dataset is downloaded and placed in the correct directory structure before running the script.

Happy Coding! 🚀

