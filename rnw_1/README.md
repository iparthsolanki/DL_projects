# Breast Cancer Classification using Deep Learning (ANN)

<div align="center">

## 📂 Project Resources

**Google Drive Notebook:**  
https://drive.google.com/file/d/1Bwwn1Rdp_Eqa4GLRLw5_6Z9JZCN6idag/view?usp=drive_link

</div>

---

# Project Overview

This project demonstrates an end-to-end **Deep Learning classification pipeline** for predicting whether a breast tumor is **Malignant** or **Benign** using an **Artificial Neural Network (ANN)** built with **TensorFlow** and **Keras**.

The notebook follows a structured learning approach, starting from data loading and preprocessing, progressing through baseline neural networks, experimenting with different optimization techniques, and finally comparing multiple models to identify the best-performing architecture.

The project also includes business insights, performance comparison, and practical recommendations for real-world healthcare applications.

---

# Business Problem

Early detection of breast cancer plays a critical role in improving patient survival rates and reducing healthcare costs. Traditional diagnosis requires expert evaluation and can sometimes be time-consuming.

By leveraging Deep Learning, healthcare professionals can build intelligent decision-support systems capable of classifying tumors accurately using medical measurements.

This project demonstrates how Artificial Neural Networks can assist in medical diagnosis while reducing prediction errors and improving decision-making.

---

# Project Objectives

- Understand the fundamentals of Artificial Neural Networks (ANN).
- Perform exploratory data analysis (EDA).
- Preprocess and standardize medical data.
- Build a baseline Single-Layer Perceptron (SLP).
- Develop Multi-Layer Perceptron (MLP) architectures.
- Compare different activation functions.
- Reduce overfitting using Early Stopping.
- Improve generalization using Dropout Layers.
- Apply L1, L2, and L1-L2 Regularization.
- Compare all Deep Learning models.
- Select the best-performing neural network.

---

# Dataset Information

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset** available in **Scikit-learn**.

### Target Classes

- Benign
- Malignant

### Dataset Characteristics

The dataset contains multiple numerical features extracted from digitized images of breast cell nuclei, including measurements related to:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

---

# Technologies Used

## Programming Language

- Python

## Data Analysis

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn

## Deep Learning

- TensorFlow
- Keras

---

# Project Workflow

## Task 1 — Data Loading & Preprocessing

Performed:

- Dataset Loading
- Dataset Inspection
- Exploratory Data Analysis (EDA)
- Class Distribution Analysis
- Correlation Heatmap
- Train-Test Split
- Feature Standardization using StandardScaler

---

## Task 2 — Single-Layer Perceptron (SLP)

Implemented a baseline neural network with:

- Single Dense Layer
- Sigmoid Activation
- Binary Classification

The SLP model serves as the benchmark for comparison with deeper architectures.

---

## Task 3 — Multi-Layer Perceptron (MLP)

Built a deeper ANN architecture by introducing hidden layers and experimenting with different activation functions.

### Activation Functions Compared

- ReLU
- Sigmoid
- Tanh

The impact of activation functions on learning performance and convergence was analyzed.

---

## Task 4 — Early Stopping

Applied **EarlyStopping Callback** to prevent unnecessary training once validation performance stopped improving.

Benefits:

- Prevents overfitting
- Reduces training time
- Improves model generalization

---

## Task 5 — Dropout Layers

Introduced Dropout layers between hidden layers to reduce neuron dependency and improve model robustness.

Advantages:

- Prevents overfitting
- Better generalization
- Improves prediction stability

---

## Task 6 — Regularization

Applied different regularization techniques to minimize model complexity.

### Techniques Used

- L1 Regularization
- L2 Regularization
- L1-L2 Regularization

These methods help reduce overfitting and improve model performance on unseen data.

---

## Task 7 — Final Model Comparison

Compared all developed models based on:

- Accuracy
- Precision
- Recall
- F1-Score
- Training History
- Validation Performance
- Loss Curves

Finally selected the best-performing ANN architecture.

---

# Deep Learning Concepts Covered

- Artificial Neural Networks (ANN)
- Single-Layer Perceptron
- Multi-Layer Perceptron
- Hidden Layers
- Activation Functions
- Forward Propagation
- Backpropagation
- Binary Classification
- Loss Functions
- Optimizers
- Early Stopping
- Dropout
- L1 Regularization
- L2 Regularization
- Model Generalization

---

# Data Preprocessing

The project includes:

- Missing Value Verification
- Class Distribution Analysis
- Feature Scaling
- Train-Test Split
- Standardization using StandardScaler

---

# Model Evaluation Metrics

The neural networks are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Training Loss
- Validation Loss

---

# Libraries Used

- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

# Skills Demonstrated

This project demonstrates practical knowledge of:

- Deep Learning
- Artificial Neural Networks
- TensorFlow
- Keras
- Data Preprocessing
- Feature Scaling
- Exploratory Data Analysis
- Model Optimization
- Early Stopping
- Dropout
- Regularization Techniques
- Classification Problems
- Medical Data Analytics

---

# Business Applications

The techniques demonstrated in this project can be applied in:

- Medical Diagnosis Systems
- Healthcare Analytics
- Clinical Decision Support
- Disease Prediction
- Hospital AI Systems
- Cancer Detection
- Healthcare Risk Assessment

---

# Project Structure

```text
Breast-Cancer-Classification-ANN/
│
├── DL_rnw1.ipynb
├── README.md
├── requirements.txt
└── Images/
```

---

# Key Outcomes

- Built an end-to-end ANN-based classification pipeline.
- Compared Single-Layer and Multi-Layer Neural Networks.
- Evaluated different activation functions.
- Reduced overfitting using Early Stopping.
- Improved model robustness with Dropout Layers.
- Applied L1, L2, and L1-L2 Regularization.
- Compared multiple Deep Learning models.
- Selected the best-performing ANN architecture.

---

# Future Improvements

- Hyperparameter Optimization
- Batch Normalization
- Learning Rate Scheduling
- TensorBoard Visualization
- SHAP Explainability
- Streamlit Deployment
- FastAPI Integration
- Docker Containerization
- Cloud Deployment (AWS, Azure, GCP)

---

# How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Breast-Cancer-Classification-ANN.git
```

### Navigate to the project

```bash
cd Breast-Cancer-Classification-ANN
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open **DL_rnw1.ipynb** and run all cells sequentially.

---

# License

This project is developed for educational and portfolio purposes.

---

# Author

## Parth Solanki

**Machine Learning Engineer | Deep Learning Enthusiast | Data Analyst**
