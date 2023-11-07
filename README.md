# Healthcare Readmission Prediction

## Overview
This project focuses on healthcare data analysis and readmission prediction. It utilizes Python libraries for data preprocessing, exploratory data analysis, and machine learning. The primary goal is to predict whether a patient will be readmitted or not.

## Project Components

### 1. Data Loading and Preprocessing
- Loads healthcare data from a CSV file.
- Handles missing values and unwanted columns.

### 2. Data Analysis
- Provides insights into patient data, such as the number of unique patients and encounters.
- Explores the distribution of readmission cases.
- Visualizes data for race, gender, age, and other factors to understand their impact.

### 3. Data Encoding
- Encodes categorical features using Label Encoding.
- Converts the 'readmitted' column into a binary classification problem.
- Drops unnecessary columns.

### 4. Machine Learning Model
- Uses a Random Forest Classifier for readmission prediction.
- Splits the data into training and testing sets.
- Evaluates model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## How to Use
1. Load your healthcare data CSV file using the `load_dataset(path)` function.
2. Use the `Preprocessing` class to clean the data and perform basic data analysis.
3. Use the `Analysis` class to gain insights into patient data.
4. Apply the `Conversion` class to encode categorical features and prepare the data for machine learning.
5. Utilize the `RFC` class to train and evaluate a Random Forest Classifier for readmission prediction.

## Dependencies
- Python 3.x
- Pandas
- Seaborn
- NumPy
- Scikit-learn
- Matplotlib

## Machine Learning Algorithm
-  Random Forest Classifier
  
## Project Structure
- `healthcare_readmission.py`: The main Python script containing the project code.
- `healthcare_data.csv`: The healthcare data file to be analyzed.

## Results
This project provides valuable insights into healthcare data and demonstrates a machine learning model for readmission prediction. It can be used for data analysis, and the provided model can be extended and fine-tuned for more advanced healthcare predictions.

Please make sure to customize the file paths and dataset according to your specific project requirements.
