"""
    NWFn(g0,m=0)

Calculates covariance matrix of sqrt(T)*sample average.

# Usage
S = NWFn(g0,m)

# Input
- `g0::Array`: Txq array of q moment conditions
- `m:int`: scalar, number of lags to use

# Output
- `S::Array`: qxq covariance matrix

"""
function NWFn(g0,m=0)

  T = size(g0,1)                    #g0 is Txq
  m = min(m,T-1)                    #number of lags

  g = g0 .- Compat.mean(g0,dims=1)  #Normalizing to Eg=0, 0.7 syntax

  S = g'g/T                         #(qxT)*(Txq)
  for s = 1:m
    Gamma_s = g[s+1:T,:]'g[1:T-s,:]/T   #same as Sum[g(t)*g(t-s)',t=s+1,T]
    S       = S  +  ( 1 - s/(m+1) ) * (Gamma_s + Gamma_s')
  end

  return S

end
