# Deep Learning – Adult Income Classification using Artificial Neural Networks

<p align="center">
  <a href="https://drive.google.com/file/d/1rG9MS1ikf7lyTwfLoyclSc6g9O__vTwc/view?usp=sharing">
    <img src="https://img.shields.io/badge/Watch%20Project%20Explanation-Google%20Drive-red?style=for-the-badge&logo=googledrive&logoColor=white" alt="Watch Project Explanation">
  </a>
</p>

<p align="center">
  <strong>Deep Learning | Artificial Neural Networks | TensorFlow | Keras | Optimization | Model Evaluation</strong>
</p>

<p align="center">
  A comprehensive deep learning project exploring ANN architecture, activation functions, weight initialization, loss functions, batch normalization, and optimization techniques for Adult Income Classification.
</p>

---

## 1. Project Overview

This project focuses on building, training, experimenting with, and evaluating **Artificial Neural Networks (ANNs)** for a binary classification problem using the Adult Income dataset.

The primary objective is to predict whether an individual's annual income is greater than $50K or less than or equal to $50K based on demographic and employment-related information.

Rather than training only one neural network, this project investigates how different deep learning components influence model learning, convergence, and classification performance.

The notebook follows a structured experimental approach, starting with data preprocessing and a baseline ANN, followed by comparisons of activation functions, weight initialization techniques, loss functions, Batch Normalization strategies, and optimizers.

The final combined ANN uses the configuration selected through these experiments and is evaluated using multiple classification metrics, including Accuracy, Precision, Recall, F1-score, and ROC-AUC.

**This project demonstrates practical understanding of neural network architecture, backpropagation-related training behavior, optimization strategies, and experimental model evaluation.**

## 2. Project Objectives

- Understand and implement an Artificial Neural Network for tabular data.
- Perform data cleaning and exploratory data analysis.
- Prepare numerical and categorical features for deep learning.
- Build a baseline Multi-Layer Perceptron (MLP).
- Compare different activation functions.
- Understand the effect of weight initialization.
- Experiment with different loss functions.
- Handle class imbalance using weighted loss.
- Implement and evaluate Focal Loss.
- Study the impact of Batch Normalization.
- Compare different optimizers.
- Investigate learning-rate sensitivity.
- Build a final combined ANN using the selected configuration.
- Evaluate models using classification metrics and ROC curves.

## 3. Business Problem

Income classification is a binary classification problem in which demographic and employment-related attributes are used to estimate whether an individual's annual income falls into one of two categories:

- `<=50K`
- `>50K`

Understanding the relationship between features such as age, education, occupation, working hours, and capital gains can help demonstrate how neural networks learn patterns from structured data.

This project uses the problem as a practical application for studying deep learning techniques and comparing their effects on classification performance.

The model is developed for educational and experimental purposes, not for making real-world employment, financial, or socioeconomic decisions about individuals.

## 4. Dataset Information

The project uses the **Adult Income dataset**, also known as the Adult Census Income dataset.

The dataset contains demographic and employment-related information along with a binary income target.

### Dataset Features

| Feature | Description |
|---|---|
| age | Age of the individual |
| workclass | Employment category |
| fnlwgt | Final sampling weight |
| education | Education qualification |
| educational-num | Numerical representation of education |
| marital-status | Marital status |
| occupation | Occupation category |
| relationship | Relationship category |
| race | Race category |
| gender | Gender |
| capital-gain | Capital gains |
| capital-loss | Capital losses |
| hours-per-week | Working hours per week |
| native-country | Country of origin |
| income | Target variable |

### Target Variable

The original income column contains two classes:

| Original Value | Encoded Value | Meaning |
|---|---:|---|
| `<=50K` | 0 | Income less than or equal to $50K |
| `>50K` | 1 | Income greater than $50K |

The target is converted into numerical binary labels to make it suitable for neural network training.

### Dataset Statistics

| Description | Value |
|---|---:|
| Original Records | 48,842 |
| Original Features | 14 input features + target |
| Records After Cleaning | 45,222 |
| Features After One-Hot Encoding | 96 |
| Training Samples | 36,177 |
| Testing Samples | 9,045 |
| Target Classes | 2 |

## 5. Data Cleaning and Exploratory Data Analysis

The first stage of the notebook focuses on understanding and preparing the dataset.

### 5.1 Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv("/content/adult.csv.zip")
```

Initial inspection includes:

- Dataset dimensions
- Data types
- First few records
- Column information
- Missing-value analysis

### 5.2 Whitespace Handling

Whitespace is removed from categorical columns:

```python
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()
```

This prevents inconsistent categorical values caused by leading or trailing spaces.

### 5.3 Missing-Value Treatment

The dataset uses `?` to represent missing categorical values.

These values are converted into NumPy NaN values:

```python
df.replace("?", np.nan, inplace=True)
```

Rows containing missing values are removed:

```python
df.dropna(inplace=True)
```

After cleaning, the dataset contains **45,222 records**, compared with the original 48,842 records.

### 5.4 Target Encoding

The income target is converted into binary values:

```python
df["income"] = (df["income"] == ">50K").astype(int)
```

This creates:

- `0` for income `<=50K`
- `1` for income `>50K`

### 5.5 Exploratory Data Analysis

The notebook explores the relationships between income and different demographic and employment-related variables.

The visualizations include:

| Visualization | Purpose |
|---|---|
| Age Distribution | Compares age distributions across income classes |
| Hours per Week Boxplot | Studies working-hour differences between income groups |
| Education Countplot | Examines income distribution across education levels |
| Correlation Heatmap | Visualizes relationships between numerical features |

### Key EDA Observations

**Age Distribution**

The distribution indicates differences in age patterns between the two income classes. Higher-income individuals are generally more represented in higher age ranges.

**Working Hours**

The boxplot shows differences in weekly working hours across income groups. The higher-income group generally displays higher working-hour patterns.

**Education**

Higher values of `educational-num` are more strongly represented in the `>50K` income group, suggesting that education is an informative feature.

**Correlation Analysis**

The correlation heatmap helps identify relationships among numerical features and the income target.

These observations describe patterns in the dataset and should not be interpreted as causal relationships.

## 6. Feature Engineering and Preprocessing

Neural networks require numerical input features. Therefore, categorical variables are converted into numerical representations and selected numerical variables are standardized.

### 6.1 Feature and Target Separation

```python
X = df.drop(["income"], axis=1)
y = df["income"]
```

- `X` contains the input features.
- `y` contains the binary income target.

### 6.2 One-Hot Encoding

Categorical features are transformed using Pandas:

```python
X = pd.get_dummies(X, drop_first=True)
```

After One-Hot Encoding, the feature matrix contains **96 columns**.

The `drop_first=True` option removes one reference category from each encoded categorical feature group.

### 6.3 Standard Scaling

StandardScaler is applied to the following three numerical columns in the executed notebook:

```python
numeric_columns = [
    "age",
    "educational-num",
    "hours-per-week"
]

scaler = StandardScaler()

X[numeric_columns] = scaler.fit_transform(
    X[numeric_columns]
)
```

Standardization transforms numerical features to a comparable scale.

The One-Hot Encoded binary features are not scaled.

### 6.4 Train-Test Split

The dataset is divided into training and testing subsets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

| Dataset | Records |
|---|---:|
| Training | 36,177 |
| Testing | 9,045 |

Stratified sampling preserves approximately the same target-class distribution in both subsets.

**Note:** For a production-ready preprocessing pipeline, the scaler should be fitted only on the training data and then applied to the test data. The notebook applies scaling before the train-test split, so this is an area that can be improved to prevent data leakage.

## 7. Baseline Artificial Neural Network

The first neural network is developed as a baseline model.

A Multi-Layer Perceptron is used because the Adult Income dataset contains structured tabular features.

### 7.1 ANN Architecture

The baseline architecture contains:

```text
Input Layer
    |
    v
Dense Layer (128 Neurons)
    |
    v
ReLU Activation
    |
    v
Dense Layer (64 Neurons)
    |
    v
ReLU Activation
    |
    v
Output Layer (1 Neuron)
    |
    v
Sigmoid Activation
    |
    v
Income Prediction
```

### 7.2 Model Configuration

| Parameter | Configuration |
|---|---|
| Model | Sequential ANN / MLP |
| Hidden Layers | 128, 64 |
| Activation | ReLU |
| Weight Initialization | Glorot Uniform |
| Output Neurons | 1 |
| Output Activation | Sigmoid |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Epochs | 50 |
| Batch Size | 64 |
| Validation Split | 10% |

### 7.3 Why Sigmoid?

The problem is binary classification.

A single output neuron with sigmoid activation produces a probability between 0 and 1.

```text
Sigmoid(x) = 1 / (1 + e^(-x))
```

The output probability is converted into a class using a threshold of 0.5.

```python
y_pred = (y_pred_prob > 0.5).astype(int)
```

### 7.4 Baseline Performance

The notebook reports the following baseline test metrics:

| Metric | Score |
|---|---:|
| Accuracy | 80.93% |
| Precision (Class 1) | 79.82% |
| Recall (Class 1) | 30.87% |
| F1-Score (Class 1) | 44.52% |

The baseline model achieves reasonable overall accuracy but has relatively low recall for the minority `>50K` class.

This motivates further experiments focused on learning behavior and class imbalance.

## 8. Activation Function Comparison

Activation functions introduce non-linearity into neural networks and influence how information and gradients pass through the layers.

The notebook compares four hidden-layer activation functions:

1. ReLU
2. Tanh
3. Sigmoid
4. ELU

Each model is trained and compared using validation accuracy and training behavior.

### 8.1 ReLU

Rectified Linear Unit:

```text
ReLU(x) = max(0, x)
```

Characteristics:

- Computationally simple.
- Produces zero for negative inputs.
- Helps reduce the vanishing-gradient problem compared with sigmoid in many settings.
- Can suffer from dead neurons.

### 8.2 Tanh

Hyperbolic tangent:

```text
Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

Characteristics:

- Output range is between -1 and 1.
- Zero-centred activation.
- Can suffer from saturation and vanishing gradients.

### 8.3 Sigmoid

```text
Sigmoid(x) = 1 / (1 + e^(-x))
```

Characteristics:

- Output range is between 0 and 1.
- Useful for binary classification output layers.
- Can suffer from vanishing gradients when used in deeper hidden layers.

### 8.4 ELU

Exponential Linear Unit:

```text
ELU(x) = x                  if x > 0
ELU(x) = α(e^x - 1)        if x <= 0
```

Characteristics:

- Allows negative outputs.
- Provides a non-zero gradient for negative inputs.
- Can help reduce the dead-neuron issue associated with ReLU.

### 8.5 Activation Experiments

The notebook compares:

- Training accuracy
- Validation accuracy
- Convergence behavior
- Class-1 F1-score
- Dead ReLU neuron fraction
- Gradient behavior in sigmoid hidden layers

The best activation should be identified from the actual experiment results rather than assumed in advance.

## 9. Weight Initialization Techniques

Weight initialization determines the starting values of neural network parameters.

Poor initialization can cause unstable gradients, slow learning, or symmetry problems.

The notebook compares five initialization techniques.

### 9.1 Glorot Uniform

Also known as Xavier Uniform initialization.

It uses a uniform distribution designed to maintain activation and gradient variance across layers.

### 9.2 Glorot Normal

Uses a normal distribution with variance based on the number of input and output units.

### 9.3 He Uniform

Designed for networks using ReLU-like activations.

It accounts for the reduced activation caused by ReLU.

### 9.4 He Normal

Uses a normal distribution designed to maintain suitable variance in ReLU networks.

### 9.5 Zeros Initialization

Initializes all weights to zero.

This causes a symmetry problem because neurons in the same layer begin with identical parameters and receive identical gradients.

As a result, they cannot learn distinct representations effectively.

### Comparison

| Initializer | Main Purpose |
|---|---|
| Glorot Uniform | Maintain activation variance |
| Glorot Normal | Maintain activation variance using normal distribution |
| He Uniform | Initialization for ReLU-based networks |
| He Normal | ReLU-compatible normal initialization |
| Zeros | Demonstrate the symmetry problem |

### Experimental Analysis

The notebook compares validation accuracy and convergence curves across the five initialization strategies.

It also demonstrates the behavior of the zero-initialized model, which remains close to the majority-class baseline.

Weight distributions are visualized before and after training to understand how optimization changes the parameter distributions.

## 10. Loss Function Experiments

The notebook investigates different loss functions for binary classification and class-imbalance handling.

The experiments include:

- Binary Cross-Entropy
- Mean Squared Error
- Weighted Binary Cross-Entropy
- Focal Loss

### 10.1 Binary Cross-Entropy

Binary Cross-Entropy is the standard loss function used for binary classification with sigmoid output.

```text
BCE = -[y log(p) + (1-y) log(1-p)]
```

Where:

- `y` is the actual target.
- `p` is the predicted probability.

It strongly penalizes confident incorrect predictions.

### 10.2 Mean Squared Error

The notebook also experiments with Mean Squared Error:

```text
MSE = (y - p)^2
```

MSE is primarily designed for regression problems.

Using MSE for sigmoid-based classification can produce less suitable gradient behavior, particularly when predictions saturate.

In the executed experiment, the MSE model's validation accuracy remains close to the majority-class baseline, demonstrating why the choice of loss function matters.

### 10.3 Weighted Binary Cross-Entropy

The Adult Income dataset contains fewer `>50K` samples than `<=50K` samples.

The class distribution after cleaning is:

| Class | Samples | Percentage |
|---|---:|---:|
| <=50K | 34,014 | 75.22% |
| >50K | 11,208 | 24.78% |

To address this imbalance, balanced class weights are calculated:

```python
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)
```

The notebook calculates approximately:

```text
Class 0 Weight = 0.6647
Class 1 Weight = 2.0175
```

The minority class receives a larger weight so that its errors contribute more strongly to the training objective.

This can improve minority-class recall, although it may also reduce precision.

The notebook compares the class-1 F1-score to determine whether weighted training is beneficial.

### 10.4 Focal Loss

Focal Loss is implemented as an additional experiment.

It reduces the contribution of easy examples and focuses training on difficult examples.

The implementation uses:

- `gamma = 2.0`
- `alpha = 0.25`

The notebook evaluates the resulting predictions using minority-class metrics.

In the recorded training run, the Focal Loss experiment remains close to the minority-class baseline and does not demonstrate successful learning. This is an important experimental result rather than evidence that Focal Loss is universally ineffective.

## 11. Batch Normalization

Batch Normalization is introduced to study its effect on ANN training behavior.

It normalizes intermediate activations using mini-batch statistics and applies learnable scale and shift parameters.

### 11.1 Batch Normalization Architecture

```text
Input
  |
  v
Dense (128)
  |
  v
Batch Normalization
  |
  v
ReLU
  |
  v
Dense (64)
  |
  v
Batch Normalization
  |
  v
ReLU
  |
  v
Dense (1)
  |
  v
Sigmoid
```

### 11.2 Learnable Parameters

Each Batch Normalization layer maintains:

| Parameter | Description |
|---|---|
| Gamma (γ) | Learnable scaling parameter |
| Beta (β) | Learnable shifting parameter |
| Moving Mean | Running mean statistic |
| Moving Variance | Running variance statistic |

Gamma and Beta are trainable, while moving mean and moving variance are non-trainable statistics used during inference.

### 11.3 Baseline vs Batch Normalization

The notebook compares the baseline ANN with the Batch Normalization model.

Recorded test results:

| Model | Accuracy | Class-1 F1-Score |
|---|---:|---:|
| Baseline ANN | 80.93% | 44.52% |
| BatchNorm ANN | 81.69% | 54.03% |

In this recorded run, Batch Normalization improves both test accuracy and minority-class F1-score compared with the baseline.

### 11.4 Batch Normalization Position Experiment

Two different layer arrangements are compared:

**Batch Normalization Before Activation**

```text
Dense → BatchNorm → ReLU
```

**Batch Normalization After Activation**

```text
Dense → ReLU → BatchNorm
```

The experiment compares validation accuracy curves to understand how layer ordering influences training behavior.

### 11.5 Gamma and Beta Visualization

The learned Gamma and Beta values from the first Batch Normalization layer are inspected and visualized.

This helps understand how the network learns scaling and shifting parameters for individual neurons.

## 12. Optimizer Comparison

Optimizers update neural network parameters during training.

The notebook compares five configurations while keeping the network architecture and other major training settings consistent.

### 12.1 Vanilla SGD

Stochastic Gradient Descent uses a fixed learning rate.

It updates parameters in the opposite direction of the calculated gradient.

### 12.2 SGD with Momentum

Momentum adds a velocity term to smooth updates and accelerate progress in consistent gradient directions.

```text
v = βv + lr × gradient
```

### 12.3 RMSprop

RMSprop adapts the learning rate using a running average of squared gradients.

This helps manage different gradient magnitudes across parameters.

### 12.4 Adam

Adam combines momentum-based first-moment estimation with adaptive second-moment estimation.

It is commonly used because it can provide effective convergence across a wide range of deep learning problems.

### 12.5 Adam with Explicit Parameters

The notebook also configures Adam explicitly:

```python
keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

### Optimizer Comparison Table

| Optimizer | Main Characteristic |
|---|---|
| SGD | Fixed global learning rate |
| SGD + Momentum | Momentum-based updates |
| RMSprop | Adaptive learning rates |
| Adam | First and second moment estimation |
| Adam Explicit | Adam with explicitly specified parameters |

### Convergence Experiment

The five models are trained using:

- ReLU activation
- He Normal initialization
- Batch Normalization
- Binary Cross-Entropy
- 50 epochs
- Batch size of 64
- Validation split of 10%

Validation accuracy curves are plotted to compare convergence.

The notebook also marks the first epoch at which each optimizer reaches 83% validation accuracy.

This helps compare convergence speed and validation performance.

## 13. Learning Rate Sensitivity

The notebook studies how learning-rate selection affects SGD and Adam.

### SGD Learning Rates

```python
sgd_learning_rates = [
    0.0001,
    0.001,
    0.01,
    0.1
]
```

### Adam Learning Rates

```python
adam_learning_rates = [
    0.0001,
    0.001,
    0.01
]
```

Each configuration is trained for 30 epochs.

Validation loss curves are plotted separately for SGD and Adam.

### Objective

- Understand how learning rate affects convergence.
- Identify unstable or ineffective learning-rate configurations.
- Compare the sensitivity of SGD and Adam.
- Understand why adaptive optimizers can be more robust to learning-rate selection.

The experiment demonstrates that optimizer performance depends on both the learning algorithm and its hyperparameters.

## 14. Final Combined ANN

The final model combines the selected configuration from the preceding experiments.

### Final Architecture

```text
Input Layer
    |
    v
Dense (128)
    |
    v
Batch Normalization
    |
    v
ReLU
    |
    v
Dense (64)
    |
    v
Batch Normalization
    |
    v
ReLU
    |
    v
Dense (1)
    |
    v
Sigmoid
    |
    v
Income Classification
```

### Final Model Configuration

| Parameter | Value |
|---|---|
| Hidden Layers | 128 → 64 |
| Activation | ReLU |
| Weight Initialization | He Normal |
| Batch Normalization | Enabled |
| Optimizer | Adam |
| Learning Rate | 0.001 (default Adam configuration) |
| Loss | Binary Cross-Entropy |
| Epochs | 80 |
| Batch Size | 64 |
| Validation Split | 10% |
| Classification Threshold | 0.5 |

Class weighting is selected conditionally based on whether Weighted BCE improves the class-1 F1-score compared with the baseline.

In the recorded notebook run, standard Binary Cross-Entropy was selected because Weighted BCE did not improve the baseline class-1 F1-score.

## 15. Final Model Evaluation

The final ANN is evaluated on the held-out test dataset.

### Recorded Test Performance

<details>
<summary><strong>Final Combined ANN — Evaluation Metrics</strong></summary>

| Metric | Score |
|---|---:|
| Accuracy | 83.10% |
| Precision (Class 1) | 64.16% |
| Recall (Class 1) | 72.03% |
| F1-Score (Class 1) | 67.87% |
| ROC-AUC | 89.96% |

</details>

### Understanding the Metrics

**Accuracy**

Measures the proportion of correctly classified samples.

**Precision**

Measures how many predicted `>50K` samples actually belong to the `>50K` class.

**Recall**

Measures how many actual `>50K` samples are correctly identified.

**F1-Score**

Provides a balance between Precision and Recall.

**ROC-AUC**

Measures the model's ability to distinguish between the two income classes across different classification thresholds.

### Baseline vs Final Model

| Metric | Baseline ANN | Final Combined ANN |
|---|---:|---:|
| Accuracy | 80.93% | 83.10% |
| Precision (Class 1) | 79.82% | 64.16% |
| Recall (Class 1) | 30.87% | 72.03% |
| F1-Score (Class 1) | 44.52% | 67.87% |
| ROC-AUC | Not recorded in baseline summary | 89.96% |

The final model improves minority-class Recall and F1-score substantially compared with the baseline.

However, Precision decreases, indicating that the final model identifies more actual higher-income cases while also producing more false positives.

This demonstrates why accuracy alone is not sufficient for evaluating an imbalanced classification problem.

## 16. Model Comparison and Visualizations

The notebook includes several visual comparisons to understand model behavior.

| Visualization | Purpose |
|---|---|
| Age Distribution | Understand feature distribution |
| Working Hours Boxplot | Compare income groups |
| Education Countplot | Study education and income patterns |
| Correlation Heatmap | Explore numerical relationships |
| Baseline Training Curves | Observe training and validation behavior |
| Confusion Matrix | Inspect classification errors |
| Activation Comparison | Compare four activation functions |
| Dead Neuron Check | Inspect inactive ReLU neurons |
| Gradient Flow Check | Investigate sigmoid gradient behavior |
| Initialization Comparison | Compare weight initialization techniques |
| Weight Distribution | Visualize parameter distributions |
| Loss Function Comparison | Evaluate classification loss strategies |
| BatchNorm Comparison | Study normalization effects |
| BatchNorm Position | Compare normalization ordering |
| Gamma and Beta Plots | Inspect learned normalization parameters |
| Optimizer Convergence | Compare optimizer behavior |
| Learning Rate Sensitivity | Compare learning-rate configurations |
| ROC Curves | Evaluate class discrimination |

## 17. Tools and Technologies

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing and model evaluation |
| TensorFlow | Deep learning framework |
| Keras | ANN model development |
| Google Colab | Notebook execution environment |
| Jupyter Notebook | Interactive experimentation |

## 18. Project Structure

```text
Adult-Income-Classification/
│
├── DL_RNW2.ipynb
├── adult.csv.zip
└── README.md
```

The notebook contains the complete data preparation, exploratory analysis, ANN experiments, training procedures, and evaluation.

The dataset file should be placed at the location expected by the notebook, or the CSV loading path should be updated.

## 19. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### Step 3: Open Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
DL_RNW2.ipynb
```

### Step 4: Load the Dataset

The notebook currently loads the dataset using:

```python
df = pd.read_csv("/content/adult.csv.zip")
```

If running locally, update this path to the actual dataset location.

### Step 5: Execute the Notebook

Run the notebook cells sequentially, from data loading and preprocessing through final model evaluation.

Some experiments depend on models, histories, and variables created in earlier cells.

## 20. Key Learnings

Through this project, I explored several important deep learning concepts.

- How neural networks process structured tabular data.
- Why preprocessing is essential for ANN training.
- How activation functions influence learning.
- Why weight initialization affects gradient flow.
- Why zero initialization causes symmetry problems.
- Why Binary Cross-Entropy is suitable for binary classification.
- How class weighting affects minority-class predictions.
- How Focal Loss changes the contribution of difficult examples.
- How Batch Normalization affects model training.
- How Gamma and Beta are learned.
- How different optimizers affect convergence.
- Why learning-rate selection matters.
- Why Precision, Recall, and F1-score are important for imbalanced datasets.
- How ROC-AUC evaluates class discrimination.

## 21. Limitations and Future Improvements

### Current Limitations

- The project focuses on a single Adult Income dataset.
- The notebook uses a fixed 0.5 classification threshold.
- Scaling is performed before the train-test split, which can introduce data leakage.
- Some experiments require repeated training and can be computationally expensive.
- Neural network results may vary across random initializations.
- The project is an experimental notebook rather than a deployed prediction application.

### Future Improvements

- Use a Scikit-learn Pipeline or ColumnTransformer for leakage-safe preprocessing.
- Fit StandardScaler only on training data.
- Add early stopping and learning-rate scheduling.
- Perform repeated experiments using multiple random seeds.
- Tune the classification threshold using validation data.
- Compare ANN performance with traditional ML models.
- Add hyperparameter optimization.
- Save the trained model and preprocessing objects.
- Build a Streamlit or FastAPI prediction interface.
- Add model explainability using SHAP.
- Evaluate fairness and potential bias in sensitive demographic features.

## 22. Conclusion

This project provides a practical and experimental study of Artificial Neural Networks for binary classification.

Starting from data cleaning and exploratory analysis, the notebook explores how different neural network design choices affect model performance.

The experiments cover activation functions, initialization methods, loss functions, Batch Normalization, optimizers, and learning-rate sensitivity.

The final combined ANN achieves **83.10% test accuracy, 67.87% minority-class F1-score, and 89.96% ROC-AUC** in the recorded run.

More importantly, the project demonstrates that evaluating a deep learning model requires more than looking at accuracy. Understanding minority-class performance, convergence, optimization, and training stability is equally important.

---

## Author

**Parth Solanki**

BCA (Hons) Student | Machine Learning | Deep Learning | Data Science | Generative AI

## Project Explanation Video

Watch the complete project explanation on Google Drive:

[**Watch AI/Deep Learning Project Explanation**](https://drive.google.com/file/d/1rG9MS1ikf7lyTwfLoyclSc6g9O__vTwc/view?usp=sharing)

## License

This project is intended for educational, academic, and portfolio purposes. Add a formal license file if you wish to define reuse permissions.
