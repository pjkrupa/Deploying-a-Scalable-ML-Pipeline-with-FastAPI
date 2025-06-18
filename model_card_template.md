# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model Type**: Random Forest Classifier  
- **Framework**: Scikit-learn  
- **Training Script**: `train_model.py`  
- **Author**: Peter Krupa  
- **Version**: 1.0  
- **Date**: June 2025  
- **File Size**: ~130MB (stored via DVC due to GitHub file size limitations)

The model is trained to classify whether an individual earns over $50K per year based on demographic and employment-related features.

## Intended Use

This model is intended to support income classification tasks for demographic analysis, educational purposes, or ML pipeline deployment demonstrations. It is not intended for deployment in high-stakes scenarios (e.g., loan approvals, hiring).

## Training Data

- **Source**: Udacity ML DevOps course starter kit: https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/blob/main/data/census.csv
- **Features**:
  - Categorical: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
  - Numerical: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Label**: `salary` (binary: `>50K`, `<=50K`, converted to `1` and `0` respectively)

The dataset was split into training and test sets using an 80/20 split. Data preprocessing included one-hot encoding for categorical features and label binarization.

## Evaluation Data

- **Split**: 20% of the dataset held out as a test set
- **Processed** using the same encoder and label binarizer as training
- **Slice Evaluation**: Model performance was additionally evaluated on categorical slices (e.g., by `education`, `sex`, `race`, etc.)

## Metrics

- **Overall Test Performance**:
  - Precision: 0.7622  
  - Recall: 0.6384  
  - F1 Score: 0.6948

- **Per-Slice Performance**:
  - Computed using a `performance_on_categorical_slice()` utility across all values of categorical features.
  - Output saved to `slice_output.txt`.

Metrics were chosen to balance concerns of false positives and false negatives in a binary classification setting.

## Ethical Considerations

- The model may reflect historical and societal biases present in the original data (e.g., income inequality across gender or race).
- Predicting income can be sensitive and should not be used to make decisions that could adversely impact individuals without further fairness auditing.
- The data includes potentially sensitive demographic information. Users should ensure privacy and data handling compliance.

## Caveats and Recommendations
- This model should **not** be used in production without extensive fairness and performance testing.
- The dataset may be outdated and not representative of current demographics.
- Further hyperparameter tuning and feature engineering could be carried out to improve model performance.