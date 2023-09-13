# Detecting Fraud Financial Transactions

This project is a Machine Learning Binary Classification Project to detect Fraud Financial Transactions in a dataset. Due to the difficulty in obtaining real time financial transactions data,
the models were simply trained on training dataset and tested on validation and test datasets, without being used to detect fraud in real time financial transactions.

# Main Software and Frameworks Used
1. JupyterLab / Google Colab was used as the development environment.
2. Pandas and Numpy were used for construction of datasets and data analysis/manipulation.
3. Matplotlib was used for data visualization.
4. TensorFlow and Pytorch were used as the Neural Networks to carry out binary classification of financial transactions.
5. SVM, KNN, Logistic Regression, Random Forest and XGBoost Algorithms were used as other Machine Learning Models to carry out binary classification of financial transactions.

## Datasets / Results
1. "fraud_0.1origbase.csv" was the labelled training dataset 

# Methodology

## Data Collection, Data Preprocessing and Feature Engineering
1. The labelled training data was obtained from github repository (https://github.com/juniorcl/transaction-fraud-detection) The dataset contained details for each financial transaction such as amount transacted, type of transaction, old balance, new balance etc, as well as a isFraud output label (1/0).
2. Descriptive statistics were calculated for numerical and categorical attributes, and exploratory data analysis carried out with matplotlib to identify trends in data (how numerical and categorical attributes affect output)
3. Feature Engineering carried out where one hot encoding was performed for 'type' categorical column, and numerical features normalized using StandardScaler library.
4. Feature Selection was carried out where irrelevant input features were dropped.
5. Pandas Dataframe with input features and output labels was split into training, validation and test datasets in 7:1.5:1.5 ratio.

## Model Training and Evaluation
6. 6 Different Machine Learning Models were used. Hyperparameter Tuning with cross validation was also performed for each model to identify the best set of hyperparameters.
   
   1. Support Vector Machine (SVM) Model was trained on training data, and evaluated on test data. After hyperparameter tuning, for class 1, it gave a recall of 0.47 and f1 score of 0.64 on test data.
   
   2. Logistic Regression Model was trained on training data, and evaluated on test data. After hyperparameter tuning, for class 1, it gave a recall of 0.47 and f1 score of 0.60 on test data.
   
   3. K Nearest Neighbours (KNN) Model was trained on training data, and evaluated on test data. After hyperparameter tuning, for class 1, it gave a recall of 0.42 and f1 score of 0.58 on test data.
   
   4. Random Forest Model was trained on training data, and evaluated on test data. After hyperparameter tuning, for class 1, it gave a recall of 0.75 and f1 score of 0.85 on test data.
   
   5. XGBoost Model was trained on training data, and evaluated on test data. After hyperparameter tuning, for class 1, it gave a recall of 0.75 and f1 score of 0.85 on test data.
   
   6. TensorFlow Neural Network Model was compiled with Adam Optimizer, Binary Cross Entropy Loss Function, and accuracy and recall metrics. StratifiedKFold object was used to carry out hyperparameter tuning with cross validation on training data. For class 1, it gave a recall of 0.67 and f1 score of 0.79 on test data.
   
   7. Pytorch Neural Network Model was compiled with Adam Optimizer and Binary Cross Entropy Loss Function. For class 1, it gave a recall of 0.618 on test data.

## Model Selection
7. The Random Forest Model was selected as it was the best model with the highest recall, f1 score and accuracy metrics, especially for class 1.
