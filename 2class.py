import numpy as np
import matplotlib.pyplot as plt


batch_size = 37
seed = 23455
rnd = np.random.RandomState(seed)

def generate():
    X = []
    Y = []
    for i in range(50):
        r = np.random.uniform(0, 3)
        X.append(r)
    for i in range(50):
        r = np.random.uniform(5, 10)
        Y.append(r)
    for i in range(50):
        r = np.random.uniform(6, 10)
        X.append(r)
    for i in range(50):
        r = np.random.uniform(0, 6)
        Y.append(r)

    R = [[]]
    for i in range(len(X)):
        R.append([X[i],Y[i]])
        #px, py = R[i][0], R[i][1]
    #R = [X,Y]
    R.remove([])
    np.random.shuffle(R)

    #X = rnd.rand(100,1)*10
    #Y = rnd.rand(100,1)*10
    X=[]
    Y=[]
    labels = []
    for i in range(len(R)):
        px, py = R[i][0], R[i][1]
        X.append(px)
        Y.append(py)
        judge = px+3-py
        if(judge > 0):
            plt.scatter(px, py, c='r')
            labels.append(1)
        elif(judge < 0):
            plt.scatter(px, py, c='b')
            labels.append(0)

    plt.xlim(-10, 30)
    plt.ylim(-10, 30)
    plt.grid()

    labels = np.array(labels)
    labels = labels.reshape(-1,1)
    X = np.array(X)
    X = X.reshape(-1,1)
    Y = np.array(Y)
    Y = Y.reshape(-1,1)
    #print(labels)
    #print(X.shape())
    #print(Y)
    return X,Y,labels

def class2():
    pass
