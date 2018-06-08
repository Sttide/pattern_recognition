from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import numpy as np
x = np.array([0, 1, 2])
y = np.array([0, 1, 2])

X = x[:, np.newaxis] # The input data for sklearn is 2D: (samples == 3 x features == 1)
print(X)



iris = load_iris()
print(iris.data.shape)
n_samples, n_features = iris.data.shape
#print(n_samples)
#print(n_features)
#print(iris.data[0])
#print(iris.target.shape)
#print(iris.target)
#print(iris.target_names)

model = LinearRegression(normalize=True)
#print(model.normalize)
#print(model)



model.fit(X, y)
print(model.coef_)
print(model.predict(np.array([3,4,5])[:, np.newaxis]))



from sklearn import neighbors, datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
print(iris.target_names[knn.predict([[3, 5, 4, 2]])]) # ['versicolor']
print(knn.score(X,y)) # 0.966666666667
print(knn.predict_proba([[3, 5, 4, 2]])) # [[ 0.   0.8  0.2]]
print(knn.predict([[3, 5, 4, 2]])) # [1]
