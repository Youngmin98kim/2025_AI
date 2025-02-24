import numpy as np

import pandas as pd


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
# print(ls)
# print(type(ls)) #pandas dataframe 객체

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root+"lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind = 'scatter', grid = True, x = "GDP per capita (USD)", y = "Life satisfaction")
plt.axis([23_500,62_500,4,9])
plt.show()

# model = LinearRegression()
# model.fit(X,y) #모델 훈련
#
# #키프로스에 대한 예측 생성
# X_new = [[37655.2]]
# print(model.predict(X_new))

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)
X_new = [[37655.2]]
print(model.predict(X_new))