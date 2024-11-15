
# CIFAR-10 Image Classification Web Application

This project classifies images from the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) and provides a **web-based frontend** for users to interact with the model. The goal is to build a web application where users can upload images, and the model will predict the class (airplane, automobile, bird, etc.).

## Key Features
- **Backend**: A CNN model built with TensorFlow/Keras for classifying CIFAR-10 images.
- **Frontend**: A simple web interface built using HTML, CSS, and JavaScript to interact with the model.
- **Data Augmentation**: The model uses data augmentation to improve generalization.
- **Visualization**: Visualize the model's training accuracy and loss over epochs.

## Dataset
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. You can download the dataset from [CIFAR-10 Website](https://www.cs.toronto.edu/~kriz/cifar.html).

## Installation

### Requirements:
- Python 3.6+
- TensorFlow 2.x
- Keras
- Flask (for backend API)
- Matplotlib
- Numpy
- HTML, CSS, and JavaScript for the frontend

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/aneetabanjara/Object-Classification-using-light-weight-CNN.git
   cd CIFAR-10-image-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   python app.py
   ```

4. Open the `index.html` file in your browser to access the web interface.

## Usage

1. **Frontend**: Open the web application in your browser. Upload an image, and the model will predict its class.

2. **Training**: To train the CNN model, run the following:
   ```bash
   python train.py
   ```
   This will save the trained model as `model.h5`.

3. **Evaluation**: To evaluate the model on the test set:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('model.h5')
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f"Test accuracy: {test_acc}")
   ```

4. **Prediction**: The frontend allows users to upload an image and receive predictions from the trained model.

## Results
The model achieves an accuracy of **X%** on the CIFAR-10 test set.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **CIFAR-10 Dataset**: Provided by CIFAR.
- **TensorFlow/Keras**: Used for model development.
- **Flask**: Used for creating the backend API to serve predictions.

---

This version includes details about the **web application** aspect, mentioning how the frontend is integrated with the backend model to allow users to interact with it via a web interface. Let me know if you need further changes!
