# ***Titanic Survival Classifier***

This project is a machine learning application built to predict passenger survival on the Titanic using two machine
learning classifiers:

1. **SVM Classifier (Custom Neural Network)**
2. **K-Nearest Neighbors (KNN) Classifier**

The classifiers are implemented in PyTorch, and data processing is handled with `pandas` and `scikit-learn`. The **SVM
Classifier** uses a fully connected neural network architecture with several dense layers and custom parameters, while
the **KNN Classifier** applies a distance-based approach for classification.

## Project Structure

- **`main.py`**: The main entry point of the application, responsible for loading data, preprocessing, model
  compilation, and training.
- **`data.py`**: Contains data preprocessing functions for cleaning, feature engineering, and creating data loaders.
- **`train.py`**: Contains functions for training the SVM and KNN models with validation support.
- **`requirements.txt`**: Lists the required packages for running the project.

## Model Descriptions

### 1. SVM Classifier (Custom Neural Network)

The SVM Classifier uses a neural network with multiple dense layers and batch normalization for binary classification.
To improve accuracy, we applied the following modifications:

- **Adjusted learning rate and optimizer**: Used `AdamW` with weight decay for regularization.
- **Custom learning rate scheduler**: Utilized `ReduceLROnPlateau` to adaptively reduce the learning rate.
- **Increased the number of epochs**: Added optional early stopping based on validation loss.
- **Experimented with loss functions**: Tried Binary Cross-Entropy Loss and Focal Loss to handle potential class
  imbalance.

### 2. K-Nearest Neighbors (KNN) Classifier

The KNN Classifier uses distance metrics to predict the survival of a passenger based on their nearest neighbors. It
supports multiple distance metrics (Euclidean and Manhattan) and allows adjusting `k` values to improve prediction
accuracy.

## Getting Started

### Prerequisites

To run this project, you’ll need Python 3.7+ and the libraries listed in `requirements.txt`.

### Running the code

To train and evaluate the models, first install the libraries, and then run main.py:

```bash
pip install -r requirements.txt 
python main.py 





