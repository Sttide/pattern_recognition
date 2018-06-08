# @Time    : 18-5-18 上午11:26
# @Email   : sttide@outlook.com

from numpy import *
import numpy as np
import matplotlib.pyplot as plt

#n = 10,100,1000,10000
n = 1000
h1 = 4

#N 窗点
N = 10000
data = np.zeros(n)
for i in range(n):
    data[i] = 0.5

x = np.linspace(0, 2, n, endpoint=True)
plt.plot(x,data,c='r',label = 'Data')

def parzen(h1,sam_N):
    print(N)
    p = np.zeros(N)
    for i in range(N):
        for j in range(sam_N):
            hn = h1 / sqrt(j+1)
            p[i] = p[i] + exp(-(2*i/N-data[j])*(2*i/N-data[j])/(2 * power(hn,2)))/(hn * sqrt(2*3.14))
        p[i] = p[i]/sam_N
    print(p.shape)
    x = np.linspace(0, 2, N, endpoint=True)
    plt.plot(x,p,c='g',label = 'Gauss')

parzen(h1, n)

def ParzenRect(h1, sam_N):
    p = np.zeros(N)
    for i in range(N):
        for j in range(sam_N):
            hn = h1 / sqrt(j+1)
            if (abs((2*i/N)-data[j])/hn)< 1/2:
                p[i] = p[i]+1
    for i in range(N):
        p[i] = 1/(N*h1)*p[i]
    x = np.linspace(0, 2, N, endpoint=True)
    plt.plot(x,p,c='b',label='Rect')
    plt.title('Parzen:h1=4,N=10000,n=1000')
    plt.legend()
    plt.show()

ParzenRect(h1, n)

print(1/(0.1938*sqrt(2*3.14)))
