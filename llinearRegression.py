import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("Salary_Data.csv")


x = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

reg= LinearRegression()

reg.fit(x_train, y_train)

y_pred =reg.predict(x_test);

mtp.scatter(x_test, y_test, color="red")
mtp.scatter(x_test, y_pred, color="blue")

mtp.title("Experience VS Salary")
mtp.xlabel("Years of Experience")
mtp.ylabel("Salary")