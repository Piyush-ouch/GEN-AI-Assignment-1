# Feedforward Neural Network from Scratch 🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Pure%20Math-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn Utility](https://img.shields.io/badge/Dataset-Iris%20Dataset-orange.svg)](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
[![Assignment Status](https://img.shields.io/badge/Assignment-Completed-brightgreen.svg)]()

A complete, ground-up implementation of a Multi-Class Feedforward Neural Network in Python **using only NumPy**. Developed as part of the **Generative AI Lab (Assignment 1)** to demonstrate the mathematical inner workings of forward propagation, backpropagation, weight initialization, loss calculation, and gradient descent optimization without relying on deep learning frameworks like TensorFlow, PyTorch, or Keras.

---

## 👨‍🎓 Student & Course Metadata

| Field | Details |
| :--- | :--- |
| **Student Name** | **Piyush Jangade and Himanshu gharde** |
| **PRN Number** | **20240111040 and 202401110007** |
| **Batch** | **A1** |
| **Course** | **Generative AI Lab** |
| **Department** | **CSE AIML** |
| **Class** | **T.Y. Tech** |
| **Date of Submission** | **15-08-2026** |

---

## 📌 Project Overview

Understanding deep learning requires building neural networks from first principles. This repository contains a fully working Jupyter Notebook (`Piyush_Jangade_GenerativeAILabAssignment.ipynb`) that implements a 3-layer neural network from scratch to solve a multi-class classification problem on the **Iris Dataset**.

### 🌟 Key Highlights
* **Zero Deep Learning Frameworks**: Built using pure vector math in **NumPy** (`np.dot`, matrix calculus).
* **Complete Mathematical Pipeline**: Manual implementation of ReLU, Softmax, Categorical Cross-Entropy, and exact analytical gradients.
* **Smart Weight Initialization**: Implements **He (Kaiming) Initialization** for ReLU hidden layers and **Xavier (Glorot) Initialization** for the Softmax output layer.
* **End-to-End Evaluation**: Includes loss curves, accuracy plots, confusion matrix generation, and precision/recall/F1 metrics.

---

## 🏗️ Neural Network Architecture

The model uses a **4 → 8 → 3** fully connected architecture:

```
[Input Layer]              [Hidden Layer]              [Output Layer]
 (4 Features)               (8 Neurons)                  (3 Classes)

  Sepal Length ────┐       ┌───────────┐                ┌───────────┐ ──> Setosa
  Sepal Width  ────┼──────>│   ReLU    │───────────────>│  Softmax  │ ──> Versicolor
  Petal Length ────┼──────>│ Activation│                │ Activation│ ──> Virginica
  Petal Width  ────┘       └───────────┘                └───────────┘
```

### 📐 Layer Configuration
* **Input Layer ($X$)**: 4 nodes corresponding to dataset features (`sepal length`, `sepal width`, `petal length`, `petal width`).
* **Hidden Layer ($H_1$)**: 8 nodes with **ReLU** non-linear activation $\text{ReLU}(z) = \max(0, z)$.
* **Output Layer ($Y$)**: 3 nodes with **Softmax** probability distribution $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}$.

---

## 🧮 Mathematical Formulation

### 1. Forward Propagation

1. **Hidden Layer Linear Transformation**:
   $$Z_1 = X \cdot W_1 + b_1$$
2. **Hidden Layer Activation**:
   $$A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$
3. **Output Layer Linear Transformation**:
   $$Z_2 = A_1 \cdot W_2 + b_2$$
4. **Output Layer Activation**:
   $$A_2 = \text{Softmax}(Z_2) = \frac{\exp(Z_2)}{\sum_{k=1}^{K} \exp(Z_2)_{:, k}}$$
5. **Categorical Cross-Entropy Loss**:
   $$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} Y_{ik} \log(A_{2, ik} + \epsilon)$$

---

### 2. Backpropagation (Analytical Gradients)

Using the chain rule of calculus, gradients are computed backwards from output to input:

1. **Output Layer Error Gradient**:
   $$dZ_2 = A_2 - Y_{\text{one-hot}}$$
2. **Output Layer Parameter Gradients**:
   $$dW_2 = \frac{1}{m} A_1^T \cdot dZ_2, \quad db_2 = \frac{1}{m} \sum_{\text{axis}=0} dZ_2$$
3. **Hidden Layer Error Gradient**:
   $$dA_1 = dZ_2 \cdot W_2^T, \quad dZ_1 = dA_1 \odot \mathbf{1}(Z_1 > 0)$$
4. **Input Layer Parameter Gradients**:
   $$dW_1 = \frac{1}{m} X^T \cdot dZ_1, \quad db_1 = \frac{1}{m} \sum_{\text{axis}=0} dZ_1$$

---

### 3. Gradient Descent Parameter Update

Parameters are updated iteratively with learning rate $\alpha$:
$$W_1 \leftarrow W_1 - \alpha \cdot dW_1, \quad b_1 \leftarrow b_1 - \alpha \cdot db_1$$
$$W_2 \leftarrow W_2 - \alpha \cdot dW_2, \quad b_2 \leftarrow b_2 - \alpha \cdot db_2$$

---

## 📂 Repository Structure

```
GEN-AI-Assignment-1/
│
├── Piyush_Jangade_GenerativeAILabAssignment.ipynb   # Main Jupyter Notebook implementation
└── README.md                                         # Detailed documentation & mathematical breakdown
```

---

## 📊 Dataset & Preprocessing

* **Dataset**: Scikit-Learn Iris Flower Dataset (150 samples, 4 features, 3 classes).
* **Train / Test Split**: 80% Training (120 samples), 20% Testing (30 samples) with fixed random seed ($42$).
* **Feature Scaling**: Standardized via Z-score normalization computed strictly on training data:
  $$\mu = \frac{1}{m}\sum X_{\text{train}}, \quad \sigma = \sqrt{\frac{1}{m}\sum (X_{\text{train}} - \mu)^2}, \quad X_{\text{scaled}} = \frac{X - \mu}{\sigma + 1e-8}$$

---

## 🚀 How to Run

### Prerequisites
Make sure Python 3.8+ and Jupyter Notebook / JupyterLab are installed along with basic dependencies:

```bash
pip install numpy matplotlib scikit-learn jupyter
```

### Execution Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Piyush-ouch/GEN-AI-Assignment-1.git
   cd GEN-AI-Assignment-1
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Piyush_Jangade_GenerativeAILabAssignment.ipynb
   ```
3. Run all cells sequentially to view training dynamics, loss graphs, and final test metrics!

---

## 📈 Performance & Results

* **Training Convergence**: Loss decreases smoothly over training epochs, demonstrating stable gradient updates.
* **Test Accuracy**: Evaluated on unseen test data to verify non-overfitting generalization.
* **Metrics Reported**: Includes per-class Precision, Recall, F1-Score, and a manual Confusion Matrix.

---

## 📜 Declaration

I declare that this assignment is my original work implemented from scratch using fundamental numerical linear algebra concepts in Python. No deep learning libraries were used for building or training the neural network.

**Student Name**: Piyush Jangade  
**PRN**: 20240111040  
**Batch**: A1  
