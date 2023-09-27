# Breast Cancer Classification using Logistic Regression

This project uses the Breast Cancer Wisconsin (Diagnostic) dataset to build a machine learning model for classifying breast tumors as benign or malignant. We use logistic regression as the classification algorithm.

## Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset is available in scikit-learn and contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes various attributes computed for cell nuclei present in the image. The target variable is binary, representing whether the tumor is benign (0) or malignant (1).

## Prerequisites

- Python 3.x
- Libraries: NumPy, Matplotlib, scikit-learn

## Usage

1. Clone this repository to your local machine or download the script.
2. Install the required libraries if not already installed: `pip install numpy matplotlib scikit-learn`.
3. Run the script:

```bash
python breast_cancer_classification.py
```

## Workflow

1. Load the Breast Cancer dataset using scikit-learn.
2. Split the dataset into training and testing sets with an 80/20 split ratio.
3. Standardize the features to have a mean of 0 and standard deviation of 1.
4. Train a logistic regression model on the training data.
5. Predict labels for the test set using the trained model.
6. Calculate the accuracy of the model on the test set.
7. Plot a confusion matrix to visualize the model's performance.
8. Display a classification report showing precision, recall, and F1-score for both benign and malignant classes.

## Results

The model achieves an accuracy of 97.36% on the test set.

![image](https://github.com/Sbt07/Breast_Cancer_Analysis/assets/93910804/8a7f5a5d-cd38-4c1d-a9a4-85f968a3fcb4)


The confusion matrix provides a breakdown of true positive, true negative, false positive, and false negative values, allowing for a deeper understanding of the model's performance.

## Acknowledgments

- Dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- Inspiration: This project is for educational purposes and is inspired by the scikit-learn documentation and tutorials.
