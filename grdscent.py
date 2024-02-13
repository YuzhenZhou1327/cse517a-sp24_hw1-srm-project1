
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-06):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    i = 0
    w = w0
    loss, grad = func(w0)
    
    while i < maxiter and np.linalg.norm(grad) > tolerance:
        w_new = w - stepsize * grad
        loss_new, grad_new = func(w_new)
        
        if loss_new < loss:
            stepsize *= 1.01
            w = w_new
            grad = grad_new
            loss = loss_new
        else:
            stepsize = max(stepsize * 0.5, eps)

        i += 1
    return w
