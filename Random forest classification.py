#============================
#IMPORTING LIBRARIES=========
#============================
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
#============================
#LOADING AND SPLITTING DATA===
#============================
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
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
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")