"""
    OlsSureFn(y,x,m=1)

LS of y on x; for one n dependent variables, same regressors

# Usage
(b,res,yhat,Covb,R2a) = OlsSureFn(y,x,m)

# Input
- `y::Array`:     Txn, the n dependent variables
- `x::Array`:     Txk matrix of regressors (including deterministic ones)
- `m::Int`:       scalar, bandwidth in Newey-West

# Output
- `b::Array`:     n*kx1, regression coefficients
- `u::Array`:     Txn, residuals y - yhat
- `yhat::Array`:  Txn, fitted values x*b
- `V::Array`:     matrix, covariance matrix of sqrt(T)vec(b)
- `R2a::Number`:  n vector, R2 value

"""
function OlsSureFn(y,x,m=0)
    (T,n) = (size(y,1),size(y,2))
    k     = size(x,2)
    b     = x\y
    yhat  = x*b
    u     = y - yhat
    g     = zeros(T,n*k)
    for i = 1:n
      vv      = (1+(i-1)*k):(i*k)   #1:k,(1+k):2k,...
      g[:,vv] = x.*u[:,i]           #moment conditions for y[:,i] regression
    end
    S0    = NWFn(g,m)            #Newey-West covariance matrix
    Sxxi  = -x'x/T
    Sxx_1 = kron(Matrix(1.0I,n,n),inv(Sxxi))    #Matrix(1.0I,n,n) is identity matrix(n)
    V     = Sxx_1 * S0 * Sxx_1
    R2a   = 1.0 .- Compat.var(u,dims=1)./Compat.var(y,dims=1)  #0.7 syntax
    return b,u,yhat,V,R2a
end
