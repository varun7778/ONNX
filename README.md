# Keras to ONNX Model Conversion and Classification

This repository demonstrates how to load a pre-trained ResNet50 model from Keras, train a custom Keras model on the CIFAR-10 dataset, and convert both models to ONNX format. It also includes functionality for performing predictions using both the Keras and ONNX models and comparing their results.

## Overview

The repository includes two main tasks:
1. **Keras to ONNX Conversion with Pre-trained ResNet50**: This section focuses on loading a pre-trained ResNet50 model from Keras, saving it as both `.h5` and `.onnx` files, and performing image classification using both formats.
2. **Training a Keras Model on CIFAR-10 and Converting to ONNX**: This section involves training a custom Convolutional Neural Network (CNN) on the CIFAR-10 dataset, saving the trained model in both `.h5` and `.onnx` formats, and comparing the inference results from both models.

### Key Features:
1. **Pre-trained ResNet50**: Convert and classify images using a pre-trained ResNet50 model in Keras, and compare the results using ONNX.
2. **Custom Keras Model on CIFAR-10**: Train a custom Keras model for multiclass classification, save it, convert it to ONNX, and compare predictions from both models.
3. **Model Comparison**: Compare predictions made using Keras and ONNX models for both tasks.
4. **Visualization**: Plot training and validation loss and accuracy for the custom model on CIFAR-10.

## Requirements

To run this code, ensure you have the following dependencies installed:

```bash
pip install tensorflow onnxruntime tf2onnx numpy pillow h5py keras2onnx
```

For older versions of TensorFlow (up to 2.3.1), use `keras2onnx` for the conversion process. For newer versions (TensorFlow 2.4.4 or higher), use `tf2onnx`.

## Usage

1. **Keras to ONNX Conversion with Pre-trained ResNet50**:
   - Place an image file (e.g., `ade20k.jpg`) in the same directory as the script, or update the `img_path` variable with the path to your image.
   - Run the script to:
     - Load and preprocess the image.
     - Classify the image using a pre-trained ResNet50 model.
     - Save the Keras model as a `.h5` file.
     - Convert the Keras model to ONNX format.
     - Perform classification again using the ONNX model.

   ```bash
   python classify_image.py
   ```

   ### Output:
   After running the script, you will see predictions printed from both the Keras model and the ONNX model. Additionally, the models are saved as:
   - `ResNet50.h5` (Keras format)
   - `ResNet50.onnx` (ONNX format)

---

2. **Training and Converting CIFAR-10 Model**:
   - Run the script to:
     - Load and preprocess the CIFAR-10 dataset.
     - Train the model on CIFAR-10 for 20 epochs.
     - Save the Keras model as a `.h5` file.
     - Convert the Keras model to ONNX format.
     - Perform inference on the test set using both the Keras and ONNX models and compare the results.

   ```bash
   python cifar_train_and_convert.py
   ```

   ### Output:
   For each test image, the following will be displayed:
   ```
   Original class is: <class_name>
   Predicted class using ONNX is: <onnx_predicted_class>
   Predicted class using Keras is: <keras_predicted_class>
   Predicted probabilities for all classes using ONNX is: <onnx_probabilities>
   Predicted probabilities for all classes using Keras is: <keras_probabilities>
   ```

---

## How it Works

### 1. Keras Image Classification (ResNet50)
The script uses the pre-trained ResNet50 model from Keras, which is trained on the ImageNet dataset. It takes an input image, preprocesses it, and predicts the top 3 labels using the `decode_predictions` function.

### 2. Keras Model Training (CIFAR-10)
The custom CNN model is trained on the CIFAR-10 dataset. It consists of several convolutional, max-pooling, dropout, and fully connected layers. After training, the model is saved as a `.h5` file for future use.

### 3. Keras to ONNX Conversion
The `tf2onnx` library is used to convert the Keras model to ONNX format. The converted model is saved with a `.onnx` extension and can be used with ONNX runtimes such as `onnxruntime` for inference.

### 4. Inference with ONNX
The `onnxruntime` library is used to perform inference with the converted ONNX model. The input image is passed through the ONNX model, and the predictions are compared with the results from the Keras model.

---

## Visualizations

The following plots are generated during the training process for the CIFAR-10 model:

1. **Training and Validation Loss**: A plot showing the loss during each epoch.
2. **Training and Validation Accuracy**: A plot showing the accuracy during each epoch.

These visualizations help in understanding how well the model is performing during training and validation.

---

## File Structure

```plaintext
.
├── classify_image.py                 # Script for Keras to ONNX conversion with ResNet50
├── cifar_train_and_convert.py        # Script for training and converting CIFAR-10 model
├── cifar_model_20epochs.h5           # Trained Keras model for CIFAR-10
├── cifar10_onnx_20epochs.onnx        # Converted ONNX model for CIFAR-10
├── ResNet50.h5                       # Pre-trained ResNet50 model (Keras format)
├── ResNet50.onnx                     # Converted ResNet50 model (ONNX format)
├── README.md                         # Project documentation
└── plots/                            # Directory for saved training plots
```

---

## Conclusion

This project provides a comprehensive guide to working with Keras models, converting them to ONNX format, and performing inference using both frameworks. It covers the conversion process for both pre-trained models (ResNet50) and custom models trained on CIFAR-10, and compares the results between Keras and ONNX models.
