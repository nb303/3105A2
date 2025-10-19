import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import generateData


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

    G1 = np.zeros((n, m), dtype=float)
    G1[:, d+1:] = -np.eye(n)
    h1 = np.zeros(n, dtype=float)

    # Multiply features by labels for each sample
    G2 = np.zeros((n, m), dtype=float)
    Y = np.diag(y.flatten())
    G2[:, :d] = - Y @ X
    G2[:, d] = -y
    G2[:, d+1:] = -np.eye(n)
    h2 = -np.ones(n, dtype=float)

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


def classify(Xtest, w, w0):
    return np.sign(Xtest @ w + w0)




def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    train_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_explinear = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    # TODO: Change the following random seed to your GROUP ID
    np.random.seed(25)
    for r in range(n_runs):
        for j, gen_model in enumerate(gen_model_list):
            Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
            Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
            for i, lamb in enumerate(lamb_list):
                #ExpLinear
                w, w0 = minExpLinear(Xtrain, ytrain, lamb)
                yhat_train = classify(Xtrain,w,w0)
                yhat_test = classify(Xtest,w,w0)
                train_acc_explinear[i, j, r] = np.mean(yhat_train == ytrain)
                test_acc_explinear[i, j, r] = np.mean(yhat_test == ytest)

                #Hinge
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                yhat_train = classify(Xtrain,w,w0)
                yhat_test = classify(Xtest,w,w0)
                train_acc_hinge[i, j, r] = np.mean(yhat_train == ytrain)
                test_acc_hinge[i, j, r] = np.mean(yhat_test == ytest)

    # TODO: compute the average accuracies over runs
    train_acc_explinear_mean = np.mean(train_acc_explinear, axis=2)
    test_acc_explinear_mean = np.mean(test_acc_explinear, axis=2)
    train_acc_hinge_mean = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge_mean = np.mean(test_acc_hinge, axis=2)

    # TODO: combine accuracies (explinear and hinge)

    train_acc = np.hstack([train_acc_explinear_mean, train_acc_hinge_mean])
    test_acc = np.hstack([test_acc_explinear_mean, test_acc_hinge_mean])

    # TODO: return 4-by-6 train accuracy and 4-by-6 test accuracy
    return train_acc, test_acc