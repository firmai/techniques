# stan code
import pandas as pd
import os

gplvm_code = lambda sig_K, cov: """
functions {
    matrix cov_linear(vector[] X, real sigma, int N, int D){
        matrix[N,N] K;
        {
            matrix[N,D] x;
            for (n in 1:N)
                x[n,] = X[n]';
            K = tcrossprod(x);
        }
        return square(sigma)*K;
    }
    
    matrix cov_matern32(vector[] X, real sigma, real l, int N){
        matrix[N,N] K;
        real dist;
        for (n in 1:N)
            for (m in 1:N){
                dist = sqrt(squared_distance(X[n], X[m]) + 1e-14);
                K[n,m] = square(sigma)*(1+sqrt(3)*dist/l)*exp(-sqrt(3)*dist/l);
            }
        return K;
    }
    
    matrix cov_matern52(vector[] X, real sigma, real l, int N){
        matrix[N,N] K;
        real dist;
        for (n in 1:N)
            for (m in 1:N){
                dist = sqrt(squared_distance(X[n], X[m]) + 1e-14);
                K[n,m] = square(sigma)*(1+sqrt(5)*dist/l+5*square(dist)/(3*square(l)))*exp(-sqrt(5)*dist/l);
            }
        return K;
    }
    
    matrix cov_exp_l2(vector[] X, real sigma, real l, int N){
        matrix[N,N] K;
        real dist;
        for (n in 1:N)
            for (m in 1:N){
                dist = sqrt(squared_distance(X[n], X[m]) + 1e-14);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
    
    matrix cov_exp(vector[] X, real sigma, real l, int N){
        matrix[N,N] K;
        real dist;
        int Q = rows(X[1]);
        for (n in 1:N)
            for (m in 1:N){
                dist = 0;  //sqrt(squared_distance(X[n], X[m]) + 1e-14);
                for (i in 1:Q)
                    dist = dist + abs(X[n,i] - X[m,i]);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
}
data {
    int<lower=0> N;
    int<lower=1> D;
    int<lower=1> Q;
    matrix[N,D] Y;
}
transformed data {
    vector[N] mu = rep_vector(0, N);
}
parameters {
    vector[Q] X[N];                     // latent space
    //real<lower=0> l_K;                // kernel lengthscales - is set to 1 in our case
    //real<lower=0> sig_K[2];           // kernel stdev
    real<lower=0> %s;                   // either sig_K or sig_K[2] or sig_K[3] - kernel variance
    vector<lower=0>[N] sigma;           // observation noise ... non isotropic a la factor model
}
transformed parameters {
    matrix[N,N] L;
    real R2 = 0;
    {
        matrix[N,N] K;
    
        //K = cov_exp_quad(X, sig_K, 1.0);
        K = %s;                 //is input to the function - e.g. cov_exp(X, sig_K, 1.0, N);

        for (n in 1:N)
            K[n,n] = K[n,n] + pow(sigma[n], 2) + 1e-14;
        L = cholesky_decompose(K);
        
        R2 = sum(1 - square(sigma) ./diagonal(K) )/N;
    }
}
model {
    for (n in 1:N)
        X[n] ~ cauchy(0, 1);
        //X[n] ~ normal(0, 1);
        
    sig_K ~ normal(0, .5);
    sigma ~ normal(0, .5);
    
    for (d in 1:D) 
        col(Y,d) ~ multi_normal_cholesky(mu, L);
}
generated quantities {
    real log_likelihood = 0;   //just the log_likelihood values. without log_prior
    real R2_hat_N = 0;
    vector[N] R2_hat_vec_N;
    matrix[N,N] K = %s;        //is input to the function - e.g. cov_exp(X, sig_K, 1.0, N);
    //matrix[N,D] Y_hat;
  
    for (d in 1:D)
        log_likelihood = log_likelihood + multi_normal_cholesky_lpdf(col(Y,d) | mu, L);
        
    {
        matrix[N,N] K_noise = K;
        matrix[N,D] Y_hat;
        matrix[N,D] resid;
        
        for (n in 1:N)
            K_noise[n,n] = K_noise[n,n] + pow(sigma[n], 2);

        Y_hat = K * mdivide_left_spd(K_noise, Y);

        resid = Y - Y_hat;
        for (n in 1:N)
            R2_hat_vec_N[n] = 1 - sum( square(row(resid,n)) )/ sum( square(row(Y,n)-mean(row(Y,n))) );
        
        R2_hat_N = mean(R2_hat_vec_N);
        //K = K_noise;           //includes non-isotropic noise to the output K 
    }
    //print(R2_hat_N);
}
""" % (sig_K, cov, cov)
#model = pystan.StanModel(model_code=gplvm_code)



def get_stan_code(kernel):
    """
        Returns stan_code for GP-LVM with input N, D, Q, Y in R^(NxD).
        Inputs:
            kernel: list of strings with max len < 4. i.e. ['linear', 'squared_exp']
                    Possible strings: 'linear', 'exp', 'squared_exp', 'matern32', 'matern52'
        Output:
            string of GP-LVM stan code
    """
    
    if len(kernel) > 3:
        print('length of kernel must be smaller then 4')
        return None

    kernel_code = {
        'squared_exp': lambda i: 'cov_exp_quad(X, {}, 1.0)'.format(i),
        'linear': lambda i: 'cov_linear(X, {}, N, Q)'.format(i),
        'matern32': lambda i: 'cov_matern32(X, {}, 1.0, N)'.format(i),
        'matern52': lambda i: 'cov_matern52(X, {}, 1.0, N)'.format(i),
        'exp': lambda i: 'cov_exp(X, {}, 1.0, N)'.format(i)
    }

    if len(kernel) == 1:
        sig_K = 'sig_K'
        cov = kernel_code[kernel[0]]('sig_K')
    if len(kernel) == 2:
        sig_K = 'sig_K[2]'
        cov = kernel_code[kernel[0]]('sig_K[1]') + ' + ' + kernel_code[kernel[1]]('sig_K[2]')
    if len(kernel) == 3:
        sig_K = 'sig_K[3]'
        cov = (kernel_code[kernel[0]]('sig_K[1]') + ' + ' + kernel_code[kernel[1]]('sig_K[2]')  +
               ' + ' + kernel_code[kernel[2]]('sig_K[3]') )

    stan_code = gplvm_code(sig_K, cov)
    return stan_code



def vb(Y, model, Q, init='random', iter=10000, tries=5, num=0):
    """
        vb - variational bayes
        Approximates the posterior with independent gaussians \
        and returns samples from the gaussians. 
    """
    N, D = Y.shape
    data_dict = {'N':N, 'D':D, 'Q':Q, 'Y':Y}
    fit = model.vb(data=data_dict, diagnostic_file='d_{}.csv'.format(num),
                   sample_file='s_{}.csv'.format(num), elbo_samples=100, init=init,
                   iter=iter)
    diagnostic = pd.read_csv('d_{}.csv'.format(num), 
                             names=['iter', 'time_in_seconds', 'ELBO'], 
                             comment='#', sep=',')
    sample = pd.read_csv('s_{}.csv'.format(num), comment='#', sep=',')
    #print('vb - ELBO: {}'.format(diagnostic.loc[:,'ELBO'].values[-1]))
    os.remove('d_{}.csv'.format(num))
    os.remove('s_{}.csv'.format(num))
    for _ in range(tries-1):
        diagnostic_, sample_ = vb(Y, model, Q, init, iter=iter, tries=1, num=num)
        if diagnostic.loc[:,'ELBO'].values[-1] < diagnostic_.loc[:,'ELBO'].values[-1]:
            diagnostic = diagnostic_
            sample = sample_
        del(diagnostic_, sample_)
    return diagnostic, sample



def optimize(Y, model, Q, init='random', iter=2000, tries=5):
    """
        returns the MAP (maximum a posteriori) estimate of the parameters.
    """
    N, D = Y.shape
    data_dict = {'N':N, 'D':D, 'Q':Q, 'Y':Y}
    opt = model.optimizing(data=data_dict, init=init, iter=iter)
    #print('opt - log_likelihood: {}'.format(opt['log_likelihood']))
    
    for _ in range(tries-1):
        opt_ = optimize(Y, model, Q, init, iter=iter, tries=1)
        if opt['log_likelihood'] < opt_['log_likelihood']:
            opt = opt_
        del(opt_)
            
    return opt
