import numpy as np
from scipy.optimize import minimize


def minExpLinear(X,y,lamb):
    n, d = X.shape
    y=y.reshape(-1)

    theta0 = np.zeros(d+1)


    def objective (theta):
        w=theta[:d]
        w0 = theta[d]
        m= y * (X.dot(w) + w0)
        loss_terms = np.maximum(0,-m) + np.exp(np.minimum(0,-m))
        return np.sum(loss_terms) + (lamb/2.0) * np.dot(w,w)

    
    res = minimize(objective, theta0, method = 'BFGS', options = {'disp': False})


    w_opt = res.x[:d].reshape(-1,1)
    w0_opt = res.x[d]

    return w_opt, w0_opt