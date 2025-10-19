import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
from A2helpers import generateData

solvers.options['show_progress'] = False

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

    np.random.seed(25)

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

def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    y = y.astype('float').reshape(-1,1)

    q = -np.ones(n) # n x 1 matrix to represent linear portion of the maximization equation, negative to convert to minimization

    G1 = -np.eye(n)
    h1 = np.zeros(n)

    G2 = np.eye(n)
    h2 = np.ones(n)

    G = np.vstack((G1, G2))  # (2n x n)
    h = np.hstack((h1, h2))  # (2n x 1)

    K = kernel_func(X, X)  # n x n
    y_delta = np.diag(y.reshape(-1))  # n x n

    P = (1/lamb) * y_delta @ K @ y_delta  # n x n
    P = P + stabilizer * np.eye(n)

    A = y.reshape(1, -1) # 1 x n

    b = np.zeros(1)  # scalar value 0

    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    A_cvx = matrix(A)
    b_cvx = matrix(b)

    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
    x = np.array(sol['x']).reshape(-1) #(n,)

    a = x.reshape(-1, 1)  # ensure its (n,1)
    
    # find closest a_i to 0.5
    closest_val = 0.5
    idx = (np.abs(a - closest_val)).argmin()

    k_i = K[idx, :].reshape(1, -1)
    y_i = y[idx]
    b_offset = (y_i - (1.0/lamb) * (k_i @ y_delta @ a)[0, 0]) # extract scalar from 1x1 array

    return a, b_offset

def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    y_diag = np.diag(y.reshape(-1))

    predictions = (1.0/lamb) * (kernel_func(Xtest, X) @ y_diag @ a) + b
    yhat_test = np.sign(predictions)
    yhat_test[yhat_test == 0] = 1

    return yhat_test.flatten()

def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1
    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])
    np.random.seed(25)

    n = X.shape[0]

    # shuffle the indices before partitioning
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k # size of each fold

    # Perform k-fold cross-validation
    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                # get the l'th fold for validation (will start at 0-fold size, then 1-fold size, etc)
                val_start = l * fold_size
                val_end = (l + 1) * fold_size if l < k - 1 else n # ensure last fold includes any remaining samples
                val_indices = indices[val_start:val_end]

                train_indices = np.setdiff1d(indices, val_indices) # get the remaining indices for training
                
                Xtrain, ytrain = X[train_indices], y[train_indices]
                Xval, yval = X[val_indices], y[val_indices]
                
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func) 
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                
                cv_acc[l, i, j] = np.mean(yval.flatten() == yhat.flatten())

    avg_cv_acc = np.mean(cv_acc, axis=0)
    
    # the lambdas are per row, different kernels per column. We want to choose the highest lambda if there is a tie
    max_acc = np.max(avg_cv_acc)
    candidates = np.argwhere(avg_cv_acc == max_acc)
    best_idx = candidates[np.argmax([lamb_list[i] for i, _ in candidates])]
    best_lamb = lamb_list[best_idx[0]]
    
    best_kernel_func = kernel_list[best_idx[1]]
    #best_kernel_idx = best_idx[1] # for testing purposes and finding which kernel we used, we return index

    return avg_cv_acc, best_lamb, best_kernel_func
