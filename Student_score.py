import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('student_scores.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print (regressor.intercept_)
print (regressor.coef_)

y_pred = regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test,'predicated':y_pred})
df
"""## Visualising the Training set results"""

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Student mark predictions(Training set)')
plt.xlabel('Hours')
plt.ylabel('Percentage mark')
plt.show()

"""## Visualising the Test set results"""
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Student Mark Prediction(Test set)')
plt.xlabel('Hours')
plt.ylabel('Percentage marks')
plt.show()
