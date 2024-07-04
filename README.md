


# Salary-Prediction
This project aims to predict the salary of employees based on their job experience, using machine learning techniques.<br>

# Prerequisites
<h3>To run this project, will need the following:<br></h3>

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

![image](https://github.com/Sanketarali/Salary-Prediction/assets/110754364/c0265108-1847-4c90-9627-7d70c7c73271)

<h3>Now before moving forward, let’s have a look at whether this dataset contains any null values or not:<br></h3>

data.isnull().sum()<br>

![image](https://github.com/Sanketarali/Salary-Prediction/assets/110754364/b35fd229-8dd8-4a24-9215-f7b785891182)

<h3>The dataset doesn’t have any null values. Let’s have a look at the relationship between the salary and job experience of the people</h3>
figure = px.scatter(data_frame = data, <br>
                    x="Salary",<br>
                    y="YearsExperience", <br>
                    size="YearsExperience", <br>
                    trendline="ols")<br>
figure.show()<br>

![image](https://github.com/Sanketarali/Salary-Prediction/assets/110754364/3267db87-2717-417f-8574-e002d763f0a5)




<h3>There is a perfect linear relationship between the salary and the job experience of the people. It means more job experience results in a higher salary.</h3><br>

# Training a Machine Learning Model


![image](https://github.com/Sanketarali/Salary-Prediction/assets/110754364/9aab263d-4fd1-44ed-aa4e-f293658b7f5c)



<h3>Now let’s predict the salary of a person using the trained Machine Learning model:</h3><br>

# Result
![image](https://github.com/Sanketarali/Salary-Prediction/assets/110754364/bf048c5d-91fc-4856-a9f1-b42b47fd8baa)



                                                


