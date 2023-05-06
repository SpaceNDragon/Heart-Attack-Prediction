

# Heart Attack Prediction

This repository contains the implementation of machine learning algorithms on the Heart Attack Dataset (heart.csv). The objective of this project is to predict the likelihood of a heart attack in a patient based on their medical attributes.

## Dataset

The Heart Attack Dataset (heart.csv) contains 303 observations and 14 attributes, including the target variable 'output', which is binary, indicating the likelihood of a heart attack (1) or not (0). The other attributes include age, sex, chest pain type, resting blood pressure, serum cholesterol levels, fasting blood sugar levels, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, slope of the peak exercise ST segment, number of major vessels (0-3) colored by fluoroscopy, and thallium stress test results.

The dataset can be downloaded from the following link:

[Heart Attack Dataset (heart.csv)](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

## Machine Learning Algorithms

The following machine learning algorithms were implemented on the Heart Attack Dataset:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

## Requirements

The following packages are required to run the code:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Clone the repository to your local machine:

```
git clone https://github.com/username/heart-attack-prediction.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```
jupyter notebook heart-attack-prediction.ipynb
```

4. Follow the instructions in the notebook to execute the machine learning algorithms.

## Results

The accuracy, precision, recall, and F1-score of each machine learning algorithm are reported in the notebook. The results show that the Random Forest algorithm achieved the highest accuracy of 88.24% on the test set. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
