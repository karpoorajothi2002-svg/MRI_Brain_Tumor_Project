# MRI Brain Tumor Detection Using Deep Learning

## Project Overview

This project is a Flask-based web application for Brain Tumor Detection using MRI images. It uses a ResNet18 deep learning model to classify MRI scans and display the prediction result through a simple web interface.

## Features

- MRI Image Upload
- Brain Tumor Detection
- Confidence Score
- Grad-CAM Visualization
- Prediction History
- Responsive User Interface

## Technologies Used

- Python
- Flask
- PyTorch
- OpenCV
- HTML
- CSS
- JavaScript

## Model

- ResNet18 (Transfer Learning)

## Project Structure

```
MRI_Brain_Tumor_Project
│
├── static/
├── templates/
├── app.py
├── gradcam.py
├── brain_tumor_resnet18_model.pth
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

## Future Improvements

- Multiple MRI slice support
- Cloud deployment
- Doctor Dashboard
- PDF Report Generation

## Author

Karpoorajothi
