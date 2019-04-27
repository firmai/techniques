import numpy as np
from cvxopt import matrix, solvers    # minimizes quadratic loss function
from scipy.optimize import minimize   # minumizes loss function with more constraints


def get_kernel_function(kernel_name):
    """
        return a kernel funtion corresponding to 'kernel_name'. 
        For stationary kernels the lengthscale is set to 1.
        input: kernel_name - string  e.g. 'linear', 'exp', 'squared_exp',
                'matern32' or 'matern52'
        output: kernel-function with input: X, kernel_variance
    """
    helper_f = lambda k, X1, X2: np.array([[k(x1,x2) for x2 in X2] for x1 in X1])
    
    def linear(X1, X2, kernel_variance):
        return kernel_variance*X1.dot(X2.T)
    
    def squared_exp(X1, X2, kernel_variance):
        k = lambda x1,x2: np.exp(-.5*np.sum( (x1-x2)**2 ))
        return kernel_variance*helper_f(k ,X1, X2)
    
    def exp(X1, X2, kernel_variance):
        k = lambda x1,x2: np.exp(-.5*np.sum( np.fabs(x1-x2) ))
        return kernel_variance*helper_f(k ,X1, X2)
    
    def matern32(X1, X2, kernel_variance):
        k = lambda x1,x2: (1+ np.sqrt(3)*np.sum( np.fabs(x1-x2) ))* \
                            np.exp(-np.sqrt(3)*np.sum( np.fabs(x1-x2) ))
        return kernel_variance*helper_f(k ,X1, X2)
    
    def matern52(X1, X2, kernel_variance):
        k = lambda x1,x2: (1+ np.sqrt(5)*np.sum(np.fabs(x1-x2))+ 5*np.sum((x1-x2)**2))* \
                            np.exp(-np.sqrt(5)*np.sum( np.fabs(x1-x2) ))
        return kernel_variance*helper_f(k ,X1, X2)
        
    kernels = {
        'linear': linear,
        'squared_exp': squared_exp,
        'exp': exp,
        'matern32': matern32,
        'matern52': matern52
    }
    
    return kernels.get(kernel_name, None)



def rand_weights(n):
    k = np.random.rand(n)
    return k/sum(k)
    
    

def error_function_minimal_variance(x, *args):  
    """
    minimizes x^T*cov*x - cov is args[0]
    """
    x = np.array(x).reshape(-1,1)
    return x.T.dot( args[0].dot(x) ).squeeze() 



def minimal_variance_portfolio(returns, cov, ret=None):
    ''' 
    input: returns - in format: number stocks x number days
    output: min(w^T*cov*w) 
       - can be extended to min(w^T*cov*w-mu*returns) [change error-function as well]
    '''
    n = len(cov)#len(returns)
    S = cov#np.cov(returns)
    
    def constraint1(x):
        return np.array(x).sum() - 1
    
    constraints = [{'type':'eq', 'fun':constraint1},]
    bounds = [(0,.1) for i in range(n)]
    
    x0 = rand_weights(n)
    outp = minimize(error_function_minimal_variance, x0, args=(S,), method='SLSQP', 
                    constraints=constraints, bounds=bounds)
    portfolio = outp['x'].reshape(-1,1)
    error2 = outp['fun']

    for i in range(10):
        x0 = rand_weights(n)
        outp = minimize(error_function_minimal_variance, x0, args=(S,), method='SLSQP', 
                        constraints=constraints, bounds=bounds, options={'ftol': 1e-9})
        error = outp['fun']
        if error < error2:
            portfolio = outp['x'].reshape(-1,1)
            error2 = error
        #print(np.sqrt(error))
    
    return np.array(portfolio).squeeze()

