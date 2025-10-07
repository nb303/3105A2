import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers


def minExpLinear(X,y,lamb):
    n, d = X.shape
    y=y.reshape(-1)


    theta0 = np.zeros(d+1)

    
    def objective (theta):
        w=theta[:d]
        w0 = theta[d]

        #Compute margin for each sample
        m= y * (X.dot(w) + w0)

        #compute ExpLinear loss for each sample
        loss_terms = np.maximum(0,-m) + np.exp(np.minimum(0,-m))

        #Total loss
        return np.sum(loss_terms) + (lamb/2.0) * np.dot(w,w)

    #Minimize loss function using BFGS
    res = minimize(objective, theta0, method = 'BFGS', options = {'disp': False})

    #Optimized weights and bias
    w_opt = res.x[:d].reshape(-1,1)
    w0_opt = res.x[d]

    return w_opt, w0_opt






solvers.options['show_progress'] = False

def minHinge(X, y, lamb, stabilizer=1e-5):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1).astype(float)
    n, d = X.shape

    # Number of optimization variables
    m = d + 1 + n


    #Build the quadratic term and only penalize w
    P = np.zeros((m, m), dtype=float)
    P[:d, :d] = lamb * np.eye(d)
    P = P + stabilizer * np.eye(m)   


    #minimize sum of slack variables
    q = np.zeros(m, dtype=float)
    q[d+1:] = 1.0  

    #Build inequality constraints
    G1 = np.zeros((n, m), dtype=float)
    #multiply features by labels for each sample
    G1[:, :d] = - (y[:, None] * X)        
    #Bias term
    G1[:, d] = -y
    #Slack variables
    G1[:, d+1:] = -np.eye(n)

    ## target values for each constraint
    h1 = -np.ones(n, dtype=float)

    #All slack variables are not negative
    G2 = np.zeros((n, m), dtype=float)
    G2[:, d+1:] = -np.eye(n)
    h2 = np.zeros(n, dtype=float)

    #Combine constraints   
    G = np.vstack([G1, G2])
    h = np.hstack([h1, h2])

    #Convert to matrices
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)

    #Solve quadratic equation
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)

    x = np.array(sol['x']).reshape(-1) 

    w = x[:d].reshape(d, 1)
    w0 = float(x[d])

    return w, w0