"""
    OlsFn(y,x,m=1)

LS of y on x; for one dependent variable

# Usage
(b,res,yhat,V,R2a) = OlsFn(y,x,m)

# Input
- `y::Array`:     Tx1, the dependent variable
- `x::Array`:     Txk matrix of regressors (including deterministic ones)
- `m::Int`:       scalar, bandwidth in Newey-West

# Output
- `b::Array`:     kx1, regression coefficients
- `u::Array`:     Tx1, residuals y - yhat
- `yhat::Array`:  Tx1, fitted values x*b
- `V::Array`:     kxk matrix, covariance matrix of sqrt(T)b
- `R2a::Number`:  scalar, R2 value

"""
function OlsFn(y,x,m=0)
    T    = size(y,1)
    b    = x\y
    yhat = x*b
    u    = y - yhat
    g    = x.*u
    S0   = NWFn(g,m)            #Newey-West covariance matrix
    Sxx  = -x'x/T
    V    = inv(Sxx)'S0*inv(Sxx)
    R2a  = 1 - var(u)/var(y)
    return b,u,yhat,V,R2a
end
