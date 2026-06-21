# Face Mask Detection System

A deep learning-based face mask detection application built using **PyTorch**, **ResNet18**, and **Streamlit**. The model classifies whether a person in an image is wearing a face mask or not and provides a confidence score for the prediction.

## Live Demo

https://facemaskdetectionresnet.streamlit.app/

## Features

* Face mask classification using a fine-tuned ResNet18 model
* Interactive web interface built with Streamlit
* Image upload and real-time prediction
* Confidence score for each prediction
* Transfer learning using ImageNet pretrained weights

## Tech Stack

* Python
* PyTorch
* TorchVision
* Streamlit
* PIL (Pillow)

## Dataset

Dataset used: Face Mask Detection Dataset

* Total Images: 7,553
* Training Images: 6,042
* Validation Images: 1,511

Classes:

* With Mask
* Without Mask

## Model Architecture

The project uses **ResNet18** pretrained on ImageNet as the base model.

### Training Pipeline

1. Load and preprocess images
2. Resize images to 224×224
3. Normalize using ImageNet statistics
4. Fine-tune ResNet18 for binary classification
5. Evaluate on a validation dataset

### Hyperparameters

* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Batch Size: 32
* Input Resolution: 224 × 224

## Results

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 99.21% |
| Validation Accuracy | 98.74% |

## Project Structure

```text
FaceMaskDetection/
│
├── app.py
├── face_mask_model_full.pth
├── requirements.txt
├── README.md
└── .gitignore
```

## Running Locally

### Clone Repository

```bash
git clone <repository-url>
cd FaceMaskDetection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

## Sample Workflow

1. Upload an image containing a person.
2. The image is preprocessed and resized.
3. The trained ResNet18 model performs inference.
4. The application displays:

   * Predicted Class
   * Confidence Score

## Current Limitations

This project is implemented as an **image classification system**, not a face detection system.

### Full-Body Images

The model performs best when the face occupies a significant portion of the image.

Performance may degrade when:

* The face is very small
* The image contains a full body
* Multiple people are present
* Background occupies most of the image

### Rotated Images

The model was trained primarily on upright faces.

Performance may decrease for:

* 90° rotated images
* Upside-down faces
* Extreme viewing angles

### Face Localization

The current implementation classifies the entire image rather than first detecting and cropping the face region.

As a result, the model may be influenced by:

* Background content
* Clothing
* Lighting conditions
* Other objects in the image

## Future Improvements

* Integrate OpenCV face detection before classification
* Support multiple faces in a single image
* Add webcam-based real-time inference
* Introduce stronger data augmentation
* Train using object detection models such as YOLOv8
* Deploy a real-time mask detection pipeline

## Learning Outcomes

Through this project I gained practical experience in:

* Deep Learning with PyTorch
* Transfer Learning
* Image Classification
* Computer Vision Workflows
* Model Evaluation
* Streamlit Deployment
* GPU-Based Training
* End-to-End Machine Learning Development
