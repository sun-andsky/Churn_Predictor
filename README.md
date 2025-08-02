# Customer Churn Prediction

## A web application built with **Streamlit** that predicts whether a customer will leave the bank using anusing a trained Machine learning model (Support Vector Machine - SVM) . 

## The app takes in customer details like credit score, age, balance, etc., and gives Results.


### Features

-  Predicts customer churn using SVM model
- Displays churn probability
- Provides retention suggestions based on profile
- Allows download of prediction report as CSV
- Scales input data using pre-fitted scaler
- Minimal, responsive UI with Streamlit

###  Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-9C27B0?style=for-the-badge&logo=python&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

---

### Main Screen

![Home Result](./Screenshots/Home_1.png)
&nbsp;&nbsp;&nbsp;&nbsp;
![Home Result](./Screenshots/Home_2.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
### Responsive design

<div >
  <img src="./Screenshots/Responsive_1.png" style="width: 30%;" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./Screenshots/Responsive_2.png" style="width: 30%;" />
</div>

---

##  Model Building Steps

1. **Data Collection**  
   Collected historical customer data from Kaggle.

2. **Exploratory Data Analysis (EDA)**  
   - Checked missing values
   - Distribution of numerical features
   - Churn ratio

3. **Data Preprocessing**  
   - Label encoded categorical variables (Gender, etc.)
   - Scaled numerical features using `StandardScaler`

4. **Feature Selection**  
   - Removed irrelevant columns like `RowNumber`, `CustomerId`, `Surname`
   - Selected top predictors using correlation and domain knowledge

5. **Model Training**  
   - Split data into train/test (80/20)
   - Used `SVC(probability=True)` from `sklearn`
   - Tuned hyperparameters with GridSearchCV

6. **Evaluation**  
   - Accuracy, Confusion Matrix, ROC-AUC Score
   - Exported model with `joblib`

7. **Deployment with Streamlit**  
   - Built interactive UI for inputs
   - Displayed prediction and suggestions
   - CSV report export functionality

---

