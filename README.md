# Machine Learning with Python Assignment

This repository contains the code for the Machine Learning with Python assignment, covering Python libraries, regression, K-NN classification and K-Means clustering.

## Project Structure

| Section | Topic | File | Dataset |
|---------|-------|------|---------|
| A | NumPy | `numpy_code.py` | None |
| A | Pandas | `pandas_code.py` | `titanic.csv` |
| A | Seaborn | `seaborn_code.py` | `iris.csv`, `titanic.csv` |
| A | Matplotlib | `matplotlib_code.py` | `life_Expentency.csv`, `iris.csv`, `titanic.csv`, `student_clustering.csv` |
| B | House Price Prediction | `house_price_linear_regression.py` | `Housing.csv` |
| B | Sentiment Analysis | `sentiment_logistic_regression.py` | `sentiment_analysis_sample_100000.csv` |
| B | Iris Species Prediction | `iris_softmax_regression.py` | `iris.csv` |
| C | K-NN on Student Data | `knn_students_classification.py` | `student_clustering.csv` |
| D | Student Clustering | `kmeans_student_clustering.py` | `student_clustering.csv` |

## Requirements

- Python 3.8 or higher
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install everything with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/arpitpaatni2007-jpg/ML_Assisgnment.git
cd ML_Assisgnment
```

2. Run any file:

```bash
python numpy_code.py
python pandas_code.py
python seaborn_code.py
python matplotlib_code.py
python house_price_linear_regression.py
python sentiment_logistic_regression.py
python iris_softmax_regression.py
python knn_students_classification.py
python kmeans_student_clustering.py
```

Keep the CSV files in the same folder as the code files. Close each chart window to let the plotting scripts continue.

## Summary of Work

### Section A: Python Libraries
- **NumPy:** arrays, indexing, slicing, broadcasting, statistics and basic linear algebra.
- **Pandas:** DataFrames, missing value cleaning, filtering, groupby, merge and reading/writing CSV.
- **Seaborn:** histogram, box, violin, scatter, heatmap, pairplot and count plot.
- **Matplotlib:** line, bar, scatter, histogram, pie and subplot charts with titles, labels and legends.

### Section B: Regression Models
- **House Price (Linear Regression):** predicts price from area, bedrooms, bathrooms, stories and parking. Reports MAE, RMSE and R2 (MAE about 1.13M, RMSE about 1.51M, R2 about 0.55).
- **Sentiment Analysis (Logistic Regression):** TF-IDF features on text reviews to classify them as positive or negative. Reports accuracy and confusion matrix.
- **Iris (Softmax Regression):** predicts the species (3 classes). Accuracy is about 93.3%.

### Section C: K-NN Classification
- Uses CGPA and IQ with scaling. The best K is chosen by testing multiple values, and the file reports accuracy and confusion matrix.

### Section D: K-Means Clustering
- Groups students by CGPA and IQ. K = 3 is chosen with the elbow method.
  - Cluster 0: high CGPA and high IQ
  - Cluster 1: low CGPA and low IQ
  - Cluster 2: medium CGPA and medium IQ

## Notes

- `Housing.csv` has no age or distance columns, so other available house features were used.
- `student_clustering.csv` has no class label, so a "placed" label was created from the combined CGPA and IQ score for the K-NN task.
- The sentiment dataset has many repeated sentences, so duplicates were removed before splitting the data.

## Author

Arpit Patni
