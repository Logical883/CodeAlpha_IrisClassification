# CodeAlpha Task 1: Iris Flower Classification

Classifying Iris flowers (setosa, versicolor, virginica) from four measurements
using scikit-learn. Part of my CodeAlpha Data Science internship.

## Dataset
Iris dataset: 150 samples, 4 features (sepal length/width, petal length/width in cm),
3 balanced classes (50 each), no missing values.

## Approach
1. Loaded and inspected the data with pandas
2. Explored it with a pairplot and correlation heatmap
3. Split 80/20 into train/test sets (stratified, random_state=42)
4. Trained a K-Nearest Neighbors classifier (k=5)
5. Evaluated with accuracy, a confusion matrix and a classification report
6. Validated with 5-fold cross-validation and compared four models

## Results
- Test accuracy (KNN): 100% (30/30)
- 5-fold cross-validation accuracy: KNN 97.3%, Logistic Regression 97.3%,
  Random Forest 96.7%, Decision Tree 95.3%
- All models perform similarly; differences are within one standard deviation.

## Key Takeaways
- Petal length and width separate the species best.
- Setosa is linearly separable; versicolor and virginica overlap slightly.
- Cross-validation gives a more reliable estimate than a single split.

## How to run
pip install -r requirements.txt
jupyter notebook iris_classification.ipynb