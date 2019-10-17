from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

X_trainp, X_testp, y_trainp, y_testp = train_test_split(X, y, test_size=0.2)
rfc = RandomForestClassifier(n_estimators=5)
rfc.fit(X_trainp, y_trainp)
y_predp = rfc.predict(X_testp)
print(accuracy_score(y_testp, y_predp))