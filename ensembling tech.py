import sklearn as sk
from sklearn.datasets import load_breast_cancer     
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.metrics import accuracy_score
#================================
#LOADING AND SPLITTING DATATSET==
#================================
iris=load_breast_cancer()
X,Y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#===================================
#FOREST OF DIFFERENT MODEL==========
#===================================
estimator= [
    ('knn',KNeighborsClassifier(n_neighbors=5,weights='distance',metric='euclidean',algorithm='kd_tree')),
    ('svm',SVC(kernel='linear',probability=True,C=1.0,random_state=42)),
    ('lr',LogisticRegression(solver='liblinear',random_state=42))
]
clf=StackingClassifier(estimators=estimator,final_estimator=RandomForestClassifier(n_estimators=100,random_state=42))
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print('accuracy:',accuracy)