# Project Report: Loan Approval Prediction Using Machine Learning

**Student:** Aditya Narayan  
**Project category:** Machine Learning — Classification

## 1. Abstract
This project demonstrates a supervised binary classification workflow for predicting a loan-status label from applicant attributes. The workflow includes data inspection, preprocessing, exploratory visualization, comparison of three classifiers, test-set evaluation, model serialization, and a Streamlit interface.

## 2. Problem Statement
Loan application assessment involves considering several applicant attributes. This academic prototype demonstrates how a classifier can learn patterns from labelled examples and return a predicted class for a new row of input data.

## 3. Objectives
- Understand supervised learning and binary classification.
- Prepare a tabular dataset for machine learning.
- Handle missing numeric and categorical values.
- Visualize selected dataset properties.
- Compare Logistic Regression, Decision Tree, and Random Forest.
- Evaluate models using accuracy, precision, recall, F1-score, and a confusion matrix.
- Save a fitted model pipeline and provide a simple interactive application.

## 4. Dataset
The included `data/loan_data.csv` has 650 synthetic rows and applicant-related input columns. The target column is `Loan_Status`, where `Y` and `N` are demonstration labels. **The dataset is synthetic and does not contain real banking records.** Its labels are generated for software demonstration; therefore, its performance metrics must not be interpreted as evidence of real-world lending accuracy.

## 5. Methodology
1. Load the CSV file using pandas.
2. Inspect columns, target labels, and missing values.
3. Generate plots for target distribution and applicant income.
4. Separate features and target.
5. Split records into 80% training and 20% testing data using stratification.
6. Impute numeric values with the median; impute categorical values with the most frequent category.
7. Scale numeric features and one-hot encode categorical features.
8. Train Logistic Regression, Decision Tree, and Random Forest pipelines.
9. Compare test accuracy, precision, recall, and F1-score for the approved class.
10. Save the selected pipeline, metadata, graphs, and text/CSV reports.
11. Use the Streamlit interface to accept inputs and display the predicted label.

## 6. Algorithms
- **Logistic Regression:** estimates class probabilities and is used as a baseline classifier.
- **Decision Tree:** learns a sequence of feature-based splits.
- **Random Forest:** combines many decision trees and aggregates their predictions.

The script selects the model with the highest approved-class F1-score on the held-out test set. Run `python train_model.py` to generate the actual results in `reports/`; do not copy unverified metric values into a final submission.

## 7. Evaluation
The script reports accuracy, precision, recall, F1-score, a classification report, and a confusion matrix. Accuracy is the proportion of correct test predictions. Precision measures the proportion of predicted approvals that were labelled approved. Recall measures the proportion of actually approved-labelled test cases found by the model. F1-score combines precision and recall. The confusion matrix shows correct and incorrect predictions by class.

## 8. Application
The Streamlit application collects applicant information and passes it to the saved preprocessing-and-classification pipeline. It displays a predicted label and, when available, estimated class probabilities. The interface is for academic demonstration only.

## 9. Limitations
- Synthetic data does not reflect real banking populations or policies.
- Test metrics on synthetic data cannot establish real-world reliability.
- A classification model may make errors and may reproduce bias present in its data.
- The prototype is not connected to bank systems and is not a real lending decision tool.

## 10. Future Improvements
- Replace synthetic data with a documented public dataset.
- Perform stronger cross-validation and hyperparameter tuning.
- Evaluate fairness and error rates across relevant groups.
- Add model explainability, input validation, and secure deployment.
- Document data provenance, assumptions, and limitations clearly.

## 11. Conclusion
The project demonstrates the main stages of a machine-learning classification application: data preparation, visualization, model comparison, evaluation, saving a fitted pipeline, and providing a user interface. The next important research step is to validate the approach using a suitable documented public dataset.
