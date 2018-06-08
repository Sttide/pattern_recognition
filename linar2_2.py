# -*- coding: utf-8 -*-
# Created Time    : 18-5-27 下午9:42
# Connect me with : sttide@outlook.com

#感知器算法

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


a = np.random.normal(1,0.2,size=(50,2))
b = np.random.normal(0,0.2,size=(50,2))

randX = np.random.normal(1,0.2,size=(50,2))
randY = np.random.normal(0,0.2,size=(50,2))
Data = np.zeros(shape=[100,3])
for i in range(50):
    Data[i] =  randX[i][0],randX[i][1], 1
for i in range(50):
    Data[50+i] = randY[i][0],randY[i][1], 0
#print(Data)
np.random.shuffle(Data)

#print(Data)
X, labels = np.array([Data[:,0], Data[:,1]]), Data[:,2]
labels = labels.reshape(-1,1)
X = X.reshape(-1,2)


clf = SVC(kernel="linear").fit(X,labels)
print(clf)
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
x1,x2 = np.meshgrid(np.arange(x1_min-1,x1_max+1,0.02),np.arange(x2_min-1,x2_max+1,0.02))
z = clf.predict(np.c_[x1.ravel(),x2.ravel()])
z = z.reshape(x1.shape)
plt.figure()
plt.contourf(x1,x2,z,cmap=plt.cm.coolwarm,alpha=0.9)
#plt.cm.coolwarm

plt.plot(a[:,0],a[:,1],'.')
plt.plot(b[:,0],b[:,1],'*')
plt.show()

