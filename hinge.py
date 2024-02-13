from numpy import maximum
import numpy as np
from checkgradHingeAndRidge import checkgradHingeAndRidge
import random

def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    margins = maximum(0, 1 - yTr * np.dot(w.T, xTr))
    loss = np.sum(margins) + lambdaa * np.sum(w ** 2)
    gradient = np.dot(-xTr, (yTr * (margins > 0)).T) + 2 * lambdaa * w
    return loss, gradient

'''
random.seed(31415926535)
    # % initial outputs
r=0
ok=0
s=[]  #used to be matlab cell array

    # data set
N=50
D=5
x=np.concatenate((np.random.randn(D,N),np.random.randn(D,N)+2),axis=1)
y=np.concatenate((np.ones((1,N)),-np.ones((1,N))),axis=1)
d = checkgradHingeAndRidge(hinge,np.random.rand(D, 1), 1e-05, x,y,10)
failtest = d > 1e-10
if failtest:
    print('fail')
else: print("pass")'''
