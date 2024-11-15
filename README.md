# CIFAR-10 Image Classification Web Application

This project classifies images from the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) and provides a **web-based frontend** built with **Flask**. Users can upload images to the web interface, and the model will predict the class of the image (e.g., airplane, automobile, bird, etc.).

## Key Features
- **Backend**: A CNN model built with TensorFlow/Keras for classifying CIFAR-10 images.
- **Frontend**: A web interface developed using **Flask** to interact with the model. 
- **Data Augmentation**: The model utilizes data augmentation techniques to improve generalization.
- **Visualization**: Visualize training accuracy and loss over epochs.

## Dataset
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. You can download the dataset from [CIFAR-10 Website](https://www.cs.toronto.edu/~kriz/cifar.html).

## Installation

### Requirements:
- Python 3.6+
- TensorFlow 2.x
- Keras
- Flask (for frontend and backend integration)
- Matplotlib
- Numpy
- HTML, CSS for styling (used in Flask templates)

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

3. Run the backend Flask server:
   ```bash
   python app.py
   ```
   The server will start, and you can access the web interface at `http://127.0.0.1:5000/`.

4. Open your browser and navigate to `http://127.0.0.1:5000/` to interact with the web application.

## Usage

1. **Frontend**: On the web interface, you can upload an image for classification. The Flask app will send the image to the backend, where the CNN model processes the image and returns the predicted class.

2. **Training**: To train the CNN model, use the following command:
   ```bash
   python train.py
   ```
   The model will be saved as `model.h5`.

3. **Evaluation**: After training the model, you can evaluate its performance on the CIFAR-10 test set:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('model.h5')
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f"Test accuracy: {test_acc}")
   ```

4. **Prediction**: The frontend allows users to upload an image, and the backend model will predict its class.

## Results
The model achieves an accuracy of **X%** on the CIFAR-10 test set.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **CIFAR-10 Dataset**: Provided by CIFAR.
- **TensorFlow/Keras**: Used for model development.
- **Flask**: Used to build the web interface and integrate with the backend model.

---

This version reflects that the frontend is built using **Flask** and also integrates Flask with the backend model. Let me know if any more adjustments are needed!
