# Keras to ONNX Model Conversion and Classification

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
pip install -r requirements.txt
```

## Usage

1. **Keras to ONNX Conversion with Pre-trained ResNet50**:
   - Place an image file (e.g., `ade20k.jpg`) in the same directory as the script, or update the `img_path` variable with the path to your image.
   - Run the script to:
     - Load and preprocess the image.
     - Classify the image using a pre-trained ResNet50 model.
     - Save the Keras model as a `.h5` file.
     - Convert the Keras model to ONNX format.
     - Perform classification again using the ONNX model.

2. **Training and Converting CIFAR-10 Model**:
   - Run the script to:
     - Load and preprocess the CIFAR-10 dataset.
     - Train the model on CIFAR-10 for 20 epochs.
     - Save the Keras model as a `.h5` file.
     - Convert the Keras model to ONNX format.
     - Perform inference on the test set using both the Keras and ONNX models and compare the results.

   ```bash
   python keras_model_to_ONNX.py
   ```

   ### Output:
   After running the script, you will see predictions printed from both the Keras model and the ONNX model. Additionally, the models are saved as:
   - `ResNet50.h5` (Keras format)
   - `ResNet50.onnx` (ONNX format)

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
