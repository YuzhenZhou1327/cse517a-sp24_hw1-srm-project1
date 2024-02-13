import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE

    logistic_loss = np.log(1 + np.exp(-yTr * np.dot(w.T, xTr) ))
    loss = np.sum(logistic_loss)
 
    gradient = xTr.dot((-yTr*(np.exp(-yTr * (w.T.dot(xTr)))/(np.exp(-yTr * (w.T.dot(xTr)))+1))).T)

    return loss,gradient
