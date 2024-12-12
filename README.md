# Load Dataset

## Description

This
 project focuses on handling and cleaning datasets for machine learning 
tasks. It includes data visualization, preprocessing, and model training
 functionalities. Key components involve the use of **scikit-learn** for model implementation and evaluation, and **matplotlib** and **seaborn** for data visualization.

## Features

- **Data Visualization:** Functions to create various plots for exploratory data analysis.
- **Data Cleaning:** Methods to preprocess and scale the data using techniques like Min-Max scaling.
- **Model Training:** Supports multiple machine learning models such as Logistic Regression, SVM, and Decision Trees.
- **Evaluation Metrics:** Includes accuracy, precision, recall, F1 score, confusion matrix, and classification reports.
- **GUI Integration:** Basic GUI functionality using **tkinter** for displaying results and user interaction.

## Dependencies

The following libraries are required for this project:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import classification_report, confusion_matrix

import joblib

from tkinter import *

from tkinter import messagebox

## Functions

### Key Functionalities

- **Visualization:** Creates plots for exploring data.
- **`dis_1` to `dis_8`:** Individual data distribution and visualization functions.
- **`Rest`:** Resets the data or GUI state.
- **`Display_Train`:** Displays training data statistics or visualizations.
- **`Display_Test`:** Displays test data statistics or visualizations.
- **`Predict_Ndata`:** Predicts outcomes on new data inputs.

## Usage

### Dataset Loading

Ensure the dataset is properly formatted and loaded into the project.

### Data Preprocessing

The project includes functionality to clean and scale the data.

### Visualization

Run the visualization functions to understand the dataset better.

### Model Training and Evaluation

- Train models like Logistic Regression, Decision Trees, and SVM.
- Evaluate their performance using the provided metrics.

### Prediction

Use the `Predict_Ndata` function to predict outcomes on unseen data.

## Getting Started

### Installation

Clone the repository and install the dependencies:

git clone <repository_url>

cd <repository_folder>

pip install -r requirements.txt

### Running the Notebook

Launch the Jupyter Notebook:

jupyter notebook Predictor.ipynb

### GUI Execution

If applicable, run the GUI for interactive data handling:

python Predictor.py

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Special thanks to the developers of Python and its scientific libraries for making such projects possible.
