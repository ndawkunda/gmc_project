# Diabetes Risk Prediction

Predicts early-stage diabetes from a patient symptom questionnaire, comparing three classic machine-learning models.

## Data

[`diabetes_data_upload.csv`](diabetes_data_upload.csv) holds 520 questionnaire responses from patients at Sylhet Diabetes Hospital in Sylhet, Bangladesh. Features are age, gender and yes/no symptoms such as polyuria, polydipsia, sudden weight loss and weakness. The target is a positive or negative diabetes diagnosis.

## Approach

The notebook, [`diabete_prediction.ipynb`](diabete_prediction.ipynb), walks through:

1. **Preprocessing:** encoding the yes/no and gender columns as 0/1 and checking for missing values.
2. **Exploration:** age distributions by diagnosis and gender, plus a feature correlation heatmap.
3. **Modeling:** each model runs in a scikit-learn pipeline with standard scaling, on an 80/20 train/test split, scored by 5-fold cross-validation.
   - Logistic regression
   - K-nearest neighbors, tuning k from 1 to 49 (best k = 5)
   - Random forest

## Results

| Model | Mean cross-validation accuracy |
| --- | --- |
| Logistic regression | 0.91 |
| Random forest | **0.97** |

Random forest was the most accurate model and was used for the final predictions.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, Jupyter

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook diabete_prediction.ipynb
```
