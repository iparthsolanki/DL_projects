# CIFAR-10 Image Classification using CNN and Transfer Learning

<p align="center">
  <a href="https://drive.google.com/file/d/1ZW0GLd_6ilDdAhorJnlWVlcIsxuatXYy/view?usp=sharing">
    <img src="https://img.shields.io/badge/Watch%20Project%20Explanation-Google%20Drive-red?style=for-the-badge&logo=googledrive&logoColor=white" alt="Watch Project Explanation Video">
  </a>
</p>

<p align="center">
  **Deep Learning | CNN | TensorFlow | Keras | CIFAR-10 | Data Augmentation | Transfer Learning**
</p>

<p align="center">
  A comprehensive deep learning project exploring Convolutional Neural Networks, image processing, feature extraction, model training, data augmentation, and MobileNetV2 transfer learning for multi-class image classification.
</p>

---

## 1. Project Overview

This project focuses on developing and comparing different Convolutional Neural Network (CNN) architectures for image classification using the CIFAR-10 dataset.

The primary objective is to classify 32×32 RGB images into ten different object categories using deep learning techniques.

The project begins with understanding image data, pixel normalization, and label encoding. It then explores the fundamental operations of CNNs, including convolution, padding, strides, pooling, and feature-map extraction.

Multiple CNN architectures are implemented, including LeNet-5 and a custom CNN built from scratch. The custom CNN is trained using backpropagation and optimized with Adam, Early Stopping, and ReduceLROnPlateau.

To improve generalization, the project experiments with two data augmentation approaches:

- ImageDataGenerator
- Keras preprocessing augmentation layers

Finally, transfer learning is implemented using ImageNet-pretrained MobileNetV2. The pretrained network is first used for feature extraction and then fine-tuned by unfreezing its top 30 layers.

The notebook also includes a six-model comparison and a demonstration of residual connections using the Functional API.

## 2. Project Objectives

- Understand the fundamentals of Convolutional Neural Networks.
- Load and explore the CIFAR-10 image dataset.
- Visualize image samples and pixel distributions.
- Understand image normalization and one-hot encoding.
- Implement manual convolution using SciPy.
- Compare VALID and SAME padding.
- Understand the effect of convolution strides.
- Visualize learned convolutional feature maps.
- Compare Max Pooling and Average Pooling.
- Understand Flatten and Global Average Pooling.
- Implement LeNet-5 architecture.
- Build a custom CNN from scratch.
- Understand forward propagation and backpropagation.
- Train CNN models using TensorFlow and Keras.
- Apply Early Stopping and learning-rate scheduling.
- Evaluate models using classification metrics.
- Implement data augmentation techniques.
- Apply transfer learning using MobileNetV2.
- Perform fine-tuning of pretrained CNN layers.
- Compare different CNN architectures and training strategies.
- Understand residual connections and their role in deep networks.

## 3. Dataset Information

The project uses the **CIFAR-10 dataset**, a widely used benchmark dataset for image classification.

CIFAR-10 contains 60,000 color images divided into ten different classes.

### Dataset Distribution

| Dataset | Images |
|---|---:|
| Training Set | 50,000 |
| Testing Set | 10,000 |
| Total Images | 60,000 |
| Image Dimensions | 32 × 32 |
| Color Channels | 3 (RGB) |
| Number of Classes | 10 |

Each image contains 32 × 32 pixels with three color channels: Red, Green, and Blue.

### CIFAR-10 Classes

The dataset contains the following ten categories:

| Class ID | Category |
|---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

### Dataset Loading

The dataset is loaded directly through TensorFlow Keras:

```python
(X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.cifar10.load_data()
```

The training and testing images are loaded separately, along with their corresponding labels.

## 4. Task A1 – Data Loading, EDA and Preprocessing

The first task focuses on understanding the image dataset and preparing it for CNN training.

### 4.1 Image Visualization

A 5×10 image grid is created to visualize five sample images from each CIFAR-10 class.

This helps understand:

- Image characteristics
- Differences between object categories
- Visual similarities between classes
- Image resolution and color information

### 4.2 Pixel Distribution Analysis

Pixel intensity distributions are visualized before and after normalization.

Original pixel values range from 0 to 255.

### 4.3 Image Normalization

The image values are converted into floating-point numbers and normalized between 0 and 1.

```python
X_train = X_train_raw.astype("float32") / 255.0
X_test = X_test_raw.astype("float32") / 255.0
```

Normalization helps maintain a consistent numerical scale during neural network training.

### 4.4 One-Hot Encoding

The class labels are converted into one-hot encoded vectors.

```python
y_train_ohe = keras.utils.to_categorical(y_train, 10)
y_test_ohe = keras.utils.to_categorical(y_test, 10)
```

For example, class 3 is represented as:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

### 4.5 CNN vs ANN

The notebook also explains why CNNs are generally more suitable than traditional fully connected ANNs for image classification.

A traditional ANN flattens an image into a one-dimensional vector, which removes the explicit spatial arrangement of pixels.

CNNs preserve local spatial information through convolution operations and use shared filter weights to identify patterns across different image locations.

## 5. Task A2 – Convolution, Padding and Strides

This task explores the fundamental convolution operation used in CNN architectures.

### 5.1 Manual Convolution

A manual edge-detection convolution is performed using SciPy's `convolve2d`.

The first CIFAR-10 image is converted into grayscale and processed using an edge-detection kernel.

```python
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)
```

The convolution highlights intensity changes and edges in the image.

### 5.2 Understanding Padding

Two common padding strategies are compared.

**VALID Padding**

- Does not add padding around the image.
- Reduces spatial dimensions after convolution.
- Uses only valid positions for filter operations.

For an input of 32×32, a 3×3 kernel, and stride 1:

```text
Output = (32 - 3) / 1 + 1
       = 30
```

Output dimensions: 30×30.

**SAME Padding**

- Adds padding around the image.
- Preserves spatial dimensions when stride is 1.
- Helps maintain image resolution through convolution layers.

Output dimensions: 32×32.

### 5.3 Understanding Strides

Stride determines how far a convolution filter moves during each operation.

The notebook compares:

- Stride = 1
- Stride = 2

Stride 1 produces denser feature maps, while stride 2 reduces spatial dimensions and computational requirements.

### 5.4 Feature Map Visualization

A Conv2D layer with 16 filters is used to generate feature maps.

The output is visualized in a 4×4 grid.

Each feature map represents the response of one convolution filter to the input image.

The visualization uses randomly initialized filters, so it demonstrates convolution mechanics rather than trained semantic features.

## 6. Task A3 – Pooling and CNN Architectures

This task explores pooling operations and implements two CNN architectures.

### 6.1 Max Pooling

MaxPooling selects the maximum activation from each local region.

Advantages:

- Retains prominent features.
- Reduces spatial dimensions.
- Helps reduce computational requirements.
- Provides some tolerance to small spatial changes.

### 6.2 Average Pooling

AveragePooling calculates the average activation within each pooling window.

It produces smoother feature maps by retaining average regional information.

### 6.3 Pooling Comparison

The notebook compares the output of:

- Original convolution feature maps
- MaxPooling
- AveragePooling

Recorded tensor shapes:

| Operation | Output Shape |
|---|---|
| Before Pooling | (1, 32, 32, 16) |
| After MaxPooling | (1, 16, 16, 16) |
| After AveragePooling | (1, 16, 16, 16) |

### 6.4 Flatten Layer

Flatten converts a multi-dimensional feature tensor into a one-dimensional vector.

The notebook demonstrates:

```text
Before Flatten: (1, 16, 16, 16)
After Flatten:  (1, 4096)
```

Flatten makes convolutional features suitable for fully connected Dense layers.

### 6.5 LeNet-5 Architecture

LeNet-5 is implemented as a classical CNN architecture adapted for CIFAR-10 RGB images.

Architecture:

```text
Input: 32 × 32 × 3
        |
        v
Conv2D (6 filters, 5×5)
        |
        v
Tanh Activation
        |
        v
Average Pooling (2×2)
        |
        v
Conv2D (16 filters, 5×5)
        |
        v
Tanh Activation
        |
        v
Average Pooling (2×2)
        |
        v
Flatten
        |
        v
Dense (120)
        |
        v
Dense (84)
        |
        v
Dense (10)
        |
        v
Softmax Output
```

LeNet-5 provides a compact baseline for understanding classical CNN architecture design.

### 6.6 Custom CNN Architecture

A modern custom CNN is developed using multiple convolution layers and Global Average Pooling.

Architecture:

```text
Input: 32 × 32 × 3
        |
        v
Conv2D (32 filters, 3×3)
        |
        v
Conv2D (32 filters, 3×3)
        |
        v
MaxPooling (2×2)
        |
        v
Conv2D (64 filters, 3×3)
        |
        v
Conv2D (64 filters, 3×3)
        |
        v
MaxPooling (2×2)
        |
        v
Conv2D (128 filters, 3×3)
        |
        v
GlobalAveragePooling2D
        |
        v
Dense (256)
        |
        v
Dense (10, Softmax)
```

### Custom CNN Parameters

<details>
<summary><strong>Model Complexity</strong></summary>

The custom CNN contains:

```text
Total Parameters: 175,018
```

The architecture uses GlobalAveragePooling2D instead of flattening the final convolutional feature maps, helping keep the parameter count manageable.

</details>

## 7. Task A4 – Backpropagation and CNN Training

This task focuses on training the custom CNN and understanding how neural networks learn through backpropagation.

### 7.1 Forward Propagation

During the forward pass, input images move through the following stages:

```text
Input Image
    |
    v
Convolution
    |
    v
Activation
    |
    v
Pooling
    |
    v
Feature Extraction
    |
    v
Global Average Pooling
    |
    v
Dense Layers
    |
    v
Softmax
    |
    v
Loss Calculation
```

### 7.2 Backpropagation

Backpropagation calculates gradients of the loss with respect to trainable parameters.

These gradients are propagated backward through the network to update convolution filters and Dense layer weights.

The notebook explains:

- Gradient flow through convolution layers.
- MaxPooling gradient behavior.
- AveragePooling gradient distribution.
- Gradient propagation through Flatten.
- Weight updates using Adam.

### 7.3 Model Compilation

The custom CNN is compiled using:

```python
model_cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### 7.4 Training Configuration

| Parameter | Configuration |
|---|---|
| Architecture | Custom CNN |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Metrics | Accuracy |
| Maximum Epochs | 100 |
| Batch Size | 64 |
| Validation Split | 10% |
| Early Stopping Patience | 10 |
| Learning Rate Reduction Factor | 0.5 |
| Minimum Learning Rate | 0.000001 |

### 7.5 Early Stopping

EarlyStopping monitors validation loss and stops training when the model stops improving.

```python
EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
```

The best-performing weights are restored after training stops.

### 7.6 ReduceLROnPlateau

ReduceLROnPlateau reduces the learning rate when validation loss stops improving.

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)
```

This helps the optimizer make smaller updates when training reaches a plateau.

### 7.7 Training Curves

The notebook visualizes:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Best validation-loss epoch
- Learning-rate reduction epochs

In the recorded run, the best validation-loss epoch was epoch 20.

### 7.8 Scratch CNN Test Performance

<details>
<summary><strong>Recorded Evaluation Results</strong></summary>

| Metric | Score |
|---|---:|
| Test Loss | 0.7472 |
| Test Accuracy | 77.88% |
| Macro Precision | 78.5% |
| Macro Recall | 77.9% |
| Macro F1-Score | 77.6% |
| Weighted F1-Score | 77.6% |

</details>

### Class-Wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Airplane | 0.844 | 0.771 | 0.806 |
| Automobile | 0.953 | 0.789 | 0.863 |
| Bird | 0.810 | 0.555 | 0.659 |
| Cat | 0.670 | 0.567 | 0.614 |
| Deer | 0.710 | 0.794 | 0.749 |
| Dog | 0.661 | 0.781 | 0.716 |
| Frog | 0.783 | 0.839 | 0.810 |
| Horse | 0.758 | 0.872 | 0.811 |
| Ship | 0.880 | 0.898 | 0.889 |
| Truck | 0.778 | 0.922 | 0.844 |

### 7.9 Confusion Matrix

A confusion matrix is generated to understand the model's classification performance across all ten classes.

It helps identify:

- Correctly classified images.
- False predictions between similar classes.
- Classes that are difficult to distinguish.
- Categories with comparatively lower recall.

### 7.10 Misclassified Image Analysis

The notebook displays ten misclassified test images with their actual and predicted labels.

This provides a visual understanding of model errors.

Similar categories such as cat/dog and automobile/truck can be difficult to distinguish at the low 32×32 image resolution.

## 8. Task A5 – Data Augmentation

Data augmentation creates modified versions of training images to improve model generalization.

Instead of relying only on the original images, the model learns from randomly transformed versions of the same images.

### 8.1 ImageDataGenerator

The first augmentation approach uses Keras ImageDataGenerator.

Configuration:

```python
ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode="nearest"
)
```

### Transformations

| Transformation | Purpose |
|---|---|
| Rotation | Introduces small angular variations |
| Width Shift | Moves image content horizontally |
| Height Shift | Moves image content vertically |
| Horizontal Flip | Creates mirrored images |
| Zoom | Changes image scale |
| Fill Mode | Fills newly created empty regions |

The notebook visualizes nine random transformations of one CIFAR-10 image.

### 8.2 CNN with ImageDataGenerator

The same custom CNN architecture is retrained using augmented image batches.

Training configuration:

- Adam optimizer
- Categorical Cross-Entropy
- Batch size of 64
- Maximum 100 epochs
- Early Stopping
- ReduceLROnPlateau

Recorded test accuracy:

```text
ImageDataGenerator CNN: 85.15%
```

### 8.3 Keras Augmentation Layers

The second augmentation approach uses modern Keras preprocessing layers.

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])
```

These transformations are integrated directly into the CNN architecture.

Recorded test accuracy:

```text
Keras Augmentation Layers CNN: 78.98%
```

### 8.4 Augmentation Comparison

| Model | Test Accuracy |
|---|---:|
| Scratch CNN without Augmentation | 77.88% |
| CNN with ImageDataGenerator | 85.15% |
| CNN with Keras Augmentation Layers | 78.98% |

In this recorded experiment, ImageDataGenerator produced the strongest test accuracy among the three scratch-CNN configurations.

The two augmentation approaches do not necessarily produce identical results because their transformations, training procedures, and validation setups differ.

**Experimental note:** The ImageDataGenerator experiment uses the first 5,000 test samples as validation data. Therefore, its reported test accuracy is not an independent evaluation against a completely untouched test set. A stricter experiment should reserve validation data from the training set and evaluate the full test set only once at the end.

## 9. Task A6 – Transfer Learning using MobileNetV2

Transfer learning reuses knowledge learned by a neural network on a large dataset and applies it to a different image classification task.

This project uses **MobileNetV2**, pretrained on ImageNet.

MobileNetV2 is a lightweight CNN architecture designed to provide efficient feature extraction.

### 9.1 Why Transfer Learning?

Training a deep CNN from scratch can require substantial training time and computational resources.

A pretrained network already contains useful visual features such as:

- Edges
- Textures
- Color patterns
- Shapes
- Object parts

These learned representations can be reused for CIFAR-10 classification.

### 9.2 Input Resizing

MobileNetV2 requires larger input dimensions than the original CIFAR-10 images.

The notebook resizes images to:

```text
96 × 96 × 3
```

A TensorFlow `tf.data` pipeline is used to resize images in batches rather than storing the entire resized dataset in memory.

### 9.3 MobileNetV2 Preprocessing

The model applies:

```python
layers.Rescaling(scale=2.0, offset=-1.0)
```

This transforms normalized pixel values from the range [0, 1] to [-1, 1], matching MobileNetV2's expected preprocessing range.

### 9.4 Feature Extraction

In the first transfer-learning stage, the pretrained MobileNetV2 base is frozen.

```python
base_model.trainable = False
```

A new classification head is added:

```text
Input (96 × 96 × 3)
        |
        v
Rescaling to [-1, 1]
        |
        v
Pretrained MobileNetV2
        |
        v
GlobalAveragePooling2D
        |
        v
Dense (128, ReLU)
        |
        v
Dense (10, Softmax)
```

Only the newly added classification head is trained.

Configuration:

| Parameter | Value |
|---|---|
| Base Model | MobileNetV2 |
| Pretrained Weights | ImageNet |
| Input Size | 96 × 96 × 3 |
| Base Model | Frozen |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | Categorical Cross-Entropy |
| Maximum Epochs | 20 |
| Batch Size | 64 |

Recorded test accuracy:

```text
MobileNetV2 Feature Extraction: 85.99%
```

### 9.5 Fine-Tuning

After feature extraction, the notebook unfreezes the top 30 layers of MobileNetV2.

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False
```

Batch Normalization layers are kept frozen for more stable fine-tuning.

A smaller learning rate is used:

```python
keras.optimizers.Adam(
    learning_rate=1e-5
)
```

The low learning rate helps adapt pretrained features without making excessively large updates to the existing weights.

Recorded test accuracy:

```text
MobileNetV2 Fine-Tuning: 87.00%
```

### 9.6 Feature Extraction vs Fine-Tuning

| Approach | Test Accuracy |
|---|---:|
| MobileNetV2 Feature Extraction | 85.99% |
| MobileNetV2 Fine-Tuning | 87.00% |

Fine-tuning improves the recorded test accuracy by approximately 1.01 percentage points compared with feature extraction.

The notebook also compares the training and validation loss and accuracy curves across both stages.

## 10. Six-Model Comparison

The notebook creates a final comparison table using six trained model configurations.

| Model | Architecture | Augmentation | Transfer Learning |
|---|---|---|---|
| LeNet-5 | Conv + Average Pool + Dense | No | No |
| CNN Scratch | Custom CNN + GAP | No | No |
| CNN + ImageDataGenerator | Custom CNN + GAP | ImageDataGenerator | No |
| CNN + Keras Augmentation | Custom CNN + GAP | Keras Layers | No |
| MobileNetV2 Feature Extraction | MobileNetV2 + Dense Head | No | Feature Extraction |
| MobileNetV2 Fine-Tuned | MobileNetV2 + Dense Head | No | Fine-Tuning |

The notebook generates the final table dynamically using the actual test accuracies and model parameter counts.

The comparison helps evaluate the trade-offs between:

- Model complexity
- Test accuracy
- Data augmentation
- Pretrained feature extraction
- Fine-tuning
- Training cost

The best model should be selected using the measured results rather than assuming that a more complex architecture always performs better.

## 11. Residual Block Demonstration

The notebook also implements a simple residual block using the Keras Functional API.

A residual connection creates a shortcut path that allows input information to bypass convolutional transformations.

### Residual Block Architecture

```text
              Input
                |
        +-------+--------+
        |                |
        v                v
    Conv2D 3×3        Conv2D 1×1
        |                |
        v                |
    Conv2D 3×3            |
        |                |
        +-------+--------+
                |
                v
             Addition
                |
                v
           ReLU Activation
                |
                v
              Output
```

### Why Residual Connections?

Residual connections help gradients flow through deeper networks by providing a shortcut around convolutional layers.

The 1×1 convolution on the shortcut path aligns the channel dimensions before addition.

Residual connections are an important concept behind architectures such as ResNet.

## 12. Model Evaluation and Visualization

The notebook includes several visual and quantitative evaluation techniques.

| Visualization | Purpose |
|---|---|
| CIFAR-10 Image Grid | Display five samples per class |
| Pixel Histograms | Compare pixel distributions before and after normalization |
| Manual Convolution | Demonstrate edge detection |
| Padding Comparison | Understand VALID and SAME padding |
| Feature Maps | Visualize convolution filters |
| Pooling Comparison | Compare MaxPooling and AveragePooling |
| Model Summaries | Inspect CNN architecture and parameters |
| Training Curves | Understand training and validation behavior |
| Confusion Matrix | Analyze class-wise predictions |
| Misclassified Images | Inspect incorrect predictions |
| Augmentation Grid | Visualize image transformations |
| Augmentation Curves | Compare training with and without augmentation |
| Transfer Learning Curves | Compare feature extraction and fine-tuning |
| Six-Model Table | Compare test accuracy and parameter counts |

## 13. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN architecture and model training |
| NumPy | Numerical operations and array processing |
| Pandas | Results analysis and comparison tables |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| SciPy | Manual convolution operation |
| Scikit-learn | Classification metrics |
| Google Colab | Notebook execution environment |
| Jupyter Notebook | Interactive development |

## 14. Project Structure

```text
CNN-CIFAR10-Image-Classification/
│
├── DL_PR4A.ipynb
├── DL_PR4A.html
├── README.md
│
└── requirements.txt
```

The notebook contains all six task groups, including data preparation, convolution experiments, pooling, CNN architecture development, model training, augmentation, transfer learning, and evaluation.

The HTML export can be included to make the notebook results easier to review without executing the code.

## 15. Installation and Execution

### Step 1: Clone the Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Open the Notebook

Open:

```text
DL_PR4A.ipynb
```

### Step 6: Execute the Notebook

Run the cells sequentially, starting from environment setup and dataset loading.

The notebook downloads CIFAR-10 through Keras and downloads ImageNet-pretrained MobileNetV2 weights when required.

**Hardware recommendation:** A GPU-enabled environment such as Google Colab is recommended because training multiple CNN architectures can take considerable time on a CPU.

## 16. Key Learnings

This project provided practical experience with:

- Image classification using deep learning.
- CNN architecture design.
- Convolution operations and filter visualization.
- Padding and stride calculations.
- MaxPooling and AveragePooling.
- Flatten and Global Average Pooling.
- LeNet-5 implementation.
- Custom CNN development.
- Forward propagation and backpropagation.
- Categorical Cross-Entropy.
- Adam optimization.
- Early Stopping and learning-rate scheduling.
- Confusion matrix and classification reports.
- Image augmentation.
- Transfer learning with MobileNetV2.
- Fine-tuning pretrained layers.
- Residual connections.
- Model comparison and performance analysis.

## 17. Limitations and Future Improvements

### Current Limitations

- CIFAR-10 images have a relatively small resolution of 32×32.
- Similar object categories can be difficult to distinguish.
- Training multiple architectures requires significant computational resources.
- Augmentation experiments use different validation procedures.
- The notebook is an experimental implementation rather than a production deployment.
- Transfer learning results depend on pretrained weights and input preprocessing.

### Future Improvements

- Experiment with ResNet50 and EfficientNet architectures.
- Add more advanced augmentation techniques.
- Implement learning-rate warm-up and cosine decay.
- Apply hyperparameter optimization.
- Add Grad-CAM for model interpretability.
- Perform more extensive class-wise error analysis.
- Save trained model weights for reuse.
- Build a Streamlit image-classification application.
- Deploy the model through FastAPI.
- Add an interface for uploading custom images.

## 18. Conclusion

This project provides a comprehensive practical study of Convolutional Neural Networks using the CIFAR-10 dataset.

Starting from image preprocessing and manual convolution, it explores the fundamental building blocks of CNNs and progresses toward custom architectures, augmentation strategies, and pretrained MobileNetV2 models.

The recorded experiments demonstrate the impact of different training strategies:

- Scratch CNN: **77.88% test accuracy**
- CNN with ImageDataGenerator: **85.15% test accuracy**
- CNN with Keras Augmentation Layers: **78.98% test accuracy**
- MobileNetV2 Feature Extraction: **85.99% test accuracy**
- MobileNetV2 Fine-Tuning: **87.00% test accuracy**

These results show how augmentation and transfer learning can improve image-classification performance, while also highlighting the importance of fair validation and independent testing.

The project demonstrates practical knowledge of deep learning fundamentals and modern CNN training techniques using TensorFlow and Keras.

---

## Project Explanation Video

Watch the complete project explanation, including the implementation and concepts covered in the notebook:

[**Watch CNN CIFAR-10 Project Explanation – Google Drive**](https://drive.google.com/file/d/1ZW0GLd_6ilDdAhorJnlWVlcIsxuatXYy/view?usp=sharing)

## Author

**Parth Solanki**

BCA (Hons) Student | Machine Learning | Deep Learning | Data Science | Generative AI

## License

This project is intended for educational, academic, and portfolio purposes. Add a formal license file if you wish to define reuse permissions.
