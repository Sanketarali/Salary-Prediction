# Salary-Prediction
This project aims to predict the salary of employees based on their job experience, using machine learning techniques.<br>

# Prerequisites
<h3>To run this project, you will need the following:<br></h3>

Python 3.x<br>
Jupyter Notebook<br>
scikit-learn library<br>
pandas library<br>
numpy library<br>

# Salary Prediction (Case Study)
 given some information about dataset like:<br>
 
 1. job experience<br>
 2. salary<br>
 
 # How  did I do?

<h3>The dataset I am using for the student marks prediction task is downloaded from Kaggle. Now let’s start with this task by importing the necessary Python libraries and dataset:<br></h3>

import pandas as pd<br>
import numpy as np<br>
import plotly.express as px<br>
import plotly.graph_objects as go<br>


data = pd.read_csv('Salary_Data.csv')<br>
data.head()<br>

![result](https://github.com/Sanketarali/Salary-Prediction/blob/main/Screenshot%20(3021).png)

<h3>Now before moving forward, let’s have a look at whether this dataset contains any null values or not:<br></h3>

data.isnull().sum()<br>

![result](https://github.com/Sanketarali/Salary-Prediction/blob/main/Screenshot%20(3022).png)

<h3>The dataset doesn’t have any null values. Let’s have a look at the relationship between the salary and job experience of the people</h3>
figure = px.scatter(data_frame = data, <br>
                    x="Salary",<br>
                    y="YearsExperience", <br>
                    size="YearsExperience", <br>
                    trendline="ols")<br>
figure.show()<br>

![result](https://github.com/Sanketarali/Salary-Prediction/blob/main/Screenshot%20(3023).png)


<h3>There is a perfect linear relationship between the salary and the job experience of the people. It means more job experience results in a higher salary.</h3><br>

# Training a Machine Learning Model
from sklearn.model_selection import train_test_split<br>
from sklearn.linear_model import LinearRegression<br>

x = np.asanyarray(data[["YearsExperience"]])<br>
y = np.asanyarray(data[["Salary"]])<br>
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)<br>

<h3>Now here’s how we can train the Machine Learning model:</h3><br>
model = LinearRegression()<br>
model.fit(xtrain, ytrain)<br>

<h3>Now let’s predict the salary of a person using the trained Machine Learning model:</h3><br>
a = float(input("Years of Experience : "))<br>
features = np.array([[a]])<br>
print("Predicted Salary = ", model.predict(features))<br>

Years of Experience : 2<br>
Predicted Salary =  [[44169.21365784]]<br>

![result](https://github.com/Sanketarali/Salary-Prediction/blob/main/Screenshot%20(3024).png)

                                                


