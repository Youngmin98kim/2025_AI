import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#연비예측모델만들기 / 데이터 정제, 결측치 처리, 분리하는 작업

mpg = sns.load_dataset("mpg").dropna()

# Feature와 Target 설정
X = mpg.drop(columns=["mpg", "name","origin"])  # 연비(MPG)를 예측할 것이므로 제외
X = pd.get_dummies(X, drop_first=True)  # 범주형 데이터 처리
y = mpg["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 회귀 계수 출력
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients.sort_values(by="Coefficient", ascending=False))

# 실제 값과 예측 값 비교 시각화
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()
#나라 이름 drop, mile for gallo
# n이 target , 분할 , 결측치 작업, 불필요한 feature drop, 수치형 데이터로 바꾸기
# test set, training set 분류
# 예측 모델 선택하기 : LinearRegression fitting
# prediction : test set에 x값을 넣음, y predict data로 나옴. y test와 비교


