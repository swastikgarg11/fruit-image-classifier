# Fruit Image Classifier

A deep learning based project that classifies fruit images using a Convolutional Neural Network (CNN).  
The model is trained on a labeled fruit dataset and can predict the fruit category from an input image.

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV  

---

## Project Features

- CNN based image classification  
- Image preprocessing and normalization  
- Model training on multiple fruit classes  
- Prediction from test images  
- Dataset included for training and testing  

---

## Dataset

The dataset contains images of different fruit categories used to train the CNN model.

Dataset structure:

dataset  
│  
├── apple  
├── banana  
├── lime  
└── orange  

Each folder contains images of the corresponding fruit class.

---

## Project Structure

fruit-image-classifier  
│  
├── dataset  
│   ├── apple  
│   ├── banana  
│   ├── lime  
│   └── orange  
│  
├── fruit_classifier.py  
├── test.jpg  
└── README.md  

---

## How to Run the Project

### 1. Install dependencies

pip install tensorflow numpy matplotlib opencv-python pillow

### 2. Run the classifier

python fruit_classifier.py

### 3. Test with an image

Use the provided test image:

test.jpg

You can also replace it with your own fruit image for testing.

---

## Model

The project uses a Convolutional Neural Network (CNN) for fruit image classification.

Main steps:

- Image preprocessing  
- Data normalization  
- CNN model training  
- Prediction on test image  

---

## Classes

The model is trained to classify the following fruit categories:

- Apple  
- Banana  
- Lime  
- Orange  

---

## Result

The CNN model achieved approximately 90% accuracy on the fruit image dataset.

---

## Example

Input Image:  
test.jpg  

Output:  
Predicted Fruit: Apple  

---

## Author

Swastik Garg  
B.Tech CSE  
SRM Institute of Science and Technology  
