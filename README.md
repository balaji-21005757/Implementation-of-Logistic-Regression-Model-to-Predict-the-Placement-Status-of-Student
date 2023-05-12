# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Load the dataset and check for null data values and duplicate data values in the dataframe.
3. Import label encoder from sklearn.preprocessing to encode the dataset.
4. Apply Logistic Regression on to the model.
5. Predict the y values.
6. Calculate the Accuracy,Confusion and Classsification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: K.Balaji
RegisterNumber: 212221230011 
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
### 1.Placement data
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/5c4b7f7e-64ca-4aa3-9ba2-2084c6c6adcf)
### 2.Salary data
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/0efe6be3-c51c-493e-b8ca-8747a91bd7e2)
### 3.Checking the null() function
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/59f82a91-ee5d-49d7-9b73-becc68d65233)
### 4. Data Duplicate
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/6cf99976-f678-49c2-84fc-c1d6cf9756ab)
### 5. Print data
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/1cd3f71f-72b4-4251-ac70-b373021ccefe)
### 6. Data-status
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/acd1eda0-803a-4cec-9411-503f59257376)
### 7. y_prediction array
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/6817489a-ad1b-427c-86c5-aefeb8974526)
### 8.Accuracy value
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/62cb71b4-952b-4644-8aa8-a4fdd7b601d4)
### 9. Confusion array
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/d9c364d5-f702-47c3-b5bf-80028019cc3c)
### 10. Classification report
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/b2097c6f-9bcc-4b43-872e-303913a3bb93)
### 11.Prediction of LR
![image](https://github.com/balaji-21005757/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/94372294/341bc5ba-de78-4708-9c5b-0ceb080c6124)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
