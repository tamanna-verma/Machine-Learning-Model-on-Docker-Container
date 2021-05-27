import pandas as pd 
import numpy as np 
import joblib as jb 
from sklearn.model import LinearRegression
dataset= pd.read_csv("SalaryData.csv")
x=dataset["YearsExperience"]
y=dataset["Salary"]
x=x.values.reshape(30,1)
model=LinearRegression()
model.fit(x,y)
years=float(input("Enter years of Experience for Salary Prediction"))
output=model.predict("[[years]]")
print(output)
