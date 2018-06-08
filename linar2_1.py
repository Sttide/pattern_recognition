#y = 2x + 2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
randX = np.random.normal(1,0.2,size=(50,2))
randY = np.random.normal(0,0.2,size=(50,2))
plt.plot(randX[:,0],randX[:,1],'.')
plt.plot(randY[:,0],randY[:,1],'*')
Data = np.zeros(shape=[100,3])
for i in range(50):
    Data[i] =  randX[i][0],randX[i][1], 0
for i in range(50):
    Data[50+i] = randY[i][0],randY[i][1], 1

np.random.shuffle(Data)
X, Y, labels = Data[:,0], Data[:,1], Data[:,2]
labels = labels.reshape(-1,1)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)



x = tf.placeholder(tf.float32,shape=(None,1))
y_ = tf.placeholder(tf.float32,shape=(None,1))
label = tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([1,2],stddev=1))
b = tf.Variable(tf.random_normal([1],stddev=1),dtype=tf.float32)


print(tf.shape(x))
print(tf.shape(w1))
y = tf.nn.sigmoid(tf.matmul(x,w1) +b )


#cross_entropy = -tf.reduce_mean(tf.log(tf.pow(y,2)))
#cross_entropy = -(label * tf.log(y)+(1-label)*tf.log(1-y))
cross_entropy = -tf.reduce_mean(tf.reduce_sum(label * tf.log(y)+(1-label)*tf.log(1-y),reduction_indices=[1]))
#loss = cross_entropy
loss = tf.reduce_sum((y-label)*(y-label))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)


with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    print("w1:\n",sess.run(w1))
    print("b:\n",sess.run(b))
    print("\n")

    STEPS=30000
    for i in range(STEPS):
        start = (i * batch_size) % 100
        end = min(start + batch_size, 100)
        _,total_loss,weight,bias,y_res,re_loss = sess.run([train_step,loss,w1,b,y,loss],feed_dict={x:X[start:end],y_:Y[start:end],label:labels[start:end]})
        if re_loss<0.001:
        	break
        if i%1000 == 0:
            # 每次选取batch_size个样本进行训练
            if i==0:
                continue
            #print("After %d training step(s),loss on all data is %s" %(i,total_loss))
            print("After %d training step(s)",i)
            print("     weight:",weight[0][0],weight[0][1])
            print("     bias:",bias[0])
            print("     loss:",re_loss)
            #print("y:",y_res)

    lx = np.linspace(-0, 1, 200)  # 生成1-10 的20[0]个点
    ly = lx * weight[0][0] + bias
    plt.plot(lx,ly, c = 'pink' )
    plt.show()





'''
真的蠢的数据生成方式
batch_size = 37
seed = 23455
rnd = np.random.RandomState(seed)

#def generate()
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
    judge = px+2-py
    if(judge > 0):
        plt.scatter(px, py, c='r')
        labels.append(1)
    elif(judge < 0):
        plt.scatter(px, py, c='b')
        labels.append(0)

plt.xlim(-10, 30)
plt.ylim(-10, 30)
plt.grid()
#plt.show()

#X = np.insert(X, 2, values=labels, axis=1)
labels = np.array(labels)
labels = labels.reshape(-1,1)
X = np.array(X)
X = X.reshape(-1,1)
Y = np.array(Y)
Y = Y.reshape(-1,1)
#print(labels)
#print(X.shape())
#print(Y)

'''
