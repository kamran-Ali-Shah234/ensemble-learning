#============================
#IMPORTING LIBRARIES=========
#============================
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
#============================
#LOADING AND SPLITTING DATA===
#============================
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
#============================
#TRAINING THE MODEL==========
#============================
model.fit(x_train, y_train)
#============================
#MAKING PREDICTIONS==========
#============================
y_pred = model.predict(x_test)
#============================
#EVALUATING THE MODEL=======
#============================
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
