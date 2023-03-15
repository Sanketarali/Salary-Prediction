import numpy as np
import pandas as pd
# import plotly.express as ps
# import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data=pd.read_csv('Salary_Data.csv')

# Create a Streamlit app
st.title('Salary Prediction App')
st.write('Enter the number of years of experience to predict the salary:')

# Add an input field for the number of years of experience
experience = st.number_input('Years of experience:', min_value=0, max_value=50)

# Create a function to predict the salary based on the input
def predict_salary(experience):
    x = np.asanyarray(data[["YearsExperience"]])
    y = np.asanyarray(data[["Salary"]])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    features = np.array([[experience]])
    prediction = model.predict(features)
    return prediction[0][0]

# Display the predicted salary
if st.button('Predict Salary'):
    prediction = predict_salary(experience)
    st.write('Predicted salary:', prediction)
