import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import linearKernel, polyKernel, gaussKernel, generateData


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

def adjExpLinear(X, y, lamb, kernel_func):
    n = X.shape[0]
    y=y.reshape(-1) # keep as shape (n,) for when I multiply it later

    alpha = np.zeros(n + 1) # n x 1 vector

    K = kernel_func(X,X)
    
    #objective function to minimize, can only take one parameter
    def objective(alpha):
        a = alpha[:n]
        a0 = alpha[n]

        m = y * (K @ a + a0)  # n x 1 vector
        loss = np.sum(np.maximum(0, -m) + np.exp(np.minimum(0, -m))) + (lamb / 2.0) * float(a.T @ K @ a) # 1 x n @ n x n = 1 x n @ n x 1 = 1 x 1
        
        return loss 
    
    res = minimize(objective, alpha, method='BFGS', options={'disp': False})

    a = res.x[:n].reshape(-1,1) #convert to explicit column vector
    a0 = res.x[n]

    return a, a0
    
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n, d = X.shape
    y = y.reshape(-1, 1) #in case

    K = kernel_func(X, X)  # n x n

    # G1 is n x (2n + 1)
    G1 = np.zeros((n, 2 * n + 1))
    G1[:n, -n:] = -np.eye(n) #first n rows and last n columns
    h1 = np.zeros(n)  # n x 1

    # G2 is n x (2n + 1)
    G2 = np.zeros((n, 2 * n + 1)) #initialize shape
    G2[:n, :n] = -np.diag(y[:, 0]) @ K  # delta y diagonal matrix multiplied by kernel
    G2[:n , n:n+1] = -y # delta y @ 1_n is just vector -y
    G2[:n, n+1:] = -np.eye(n) #negative identity for the slack variable
    h2 = -np.ones(n)  # n x 1

    G = np.vstack((G1, G2))  # (2n x (2n + 1))
    h = np.hstack((h1, h2))  # (2n x 1)

    # P matrix (2n + 1) x (2n + 1)
    P = np.zeros((2 * n + 1, 2 * n + 1))
    P[:n, :n] = lamb * K #set the top left corner to lambda * K
    P = P + stabilizer * np.eye(2 * n + 1) #add stabilizer 

    # q column vector (2n + 1) x 1 --> with n + 1 0's and n 1's
    q = np.zeros((2 * n + 1, 1))
    q[n+1:] = 1 #set the last n elements to 1

    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)

    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    x = np.array(sol['x']).reshape(-1)

    a = x[:n].reshape(n, 1) #should be n x 1
    a0 = x[n]

    return a, a0

def adjClassify(Xtest, a, a0, X, kernel_func):
    yhat_test = np.sign(kernel_func(Xtest, X) @ a + a0)
    yhat_test[yhat_test == 0] = 1  # in case prediction is exactly zero, classify as +1

    return yhat_test

def synExperimentsKernel():
    n_runs = 10 
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
    lambda X1, X2: polyKernel(X1, X2, 2),
    lambda X1, X2: polyKernel(X1, X2, 3),
    lambda X1, X2: gaussKernel(X1, X2, 1.0),
    lambda X1, X2: gaussKernel(X1, X2, 0.5)]

    gen_model_list = [1, 2, 3]
    train_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_explinear = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])

    np.random.seed(101218051)

    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                    Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                    Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                    # For training accuracy, we compute the accuracy of the training set, supported by the training set itself
                    a, a0 = adjExpLinear(Xtrain, ytrain, lamb, kernel)
                    train_acc_explinear[i, j, r] = np.mean(ytrain.reshape(-1) == adjClassify(Xtrain, a, a0, Xtrain, kernel).reshape(-1)) # make sure the shapes are consistent
                    test_acc_explinear[i, j, r] = np.mean(ytest.reshape(-1) == adjClassify(Xtest, a, a0, Xtrain, kernel).reshape(-1))
                    
                    a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                    train_acc_hinge[i, j, r] = np.mean(ytrain.reshape(-1) == adjClassify(Xtrain, a, a0, Xtrain, kernel).reshape(-1)) 
                    test_acc_hinge[i, j, r] = np.mean(ytest.reshape(-1) == adjClassify(Xtest, a, a0, Xtrain, kernel).reshape(-1))

    train_acc_explinear = np.mean(train_acc_explinear, axis=2)
    test_acc_explinear = np.mean(test_acc_explinear, axis=2)
    train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    train_acc = np.hstack((train_acc_explinear, train_acc_hinge))
    test_acc  = np.hstack((test_acc_explinear, test_acc_hinge))

    return train_acc, test_acc


print(synExperimentsKernel())
    



