# -*- coding: utf-8 -*-
"""
Python module to solve for and analyze full commitment and renegotiation-proof
contracts.  See https://github.com/jhconning/renegotiation
Note: code originally had delta and r... but some formulas assume delta=1/(1+r)

For a paper by Karna Basu and Jonathan Conning

@author: Jonathan Conning
"""
import numpy as np
from scipy.optimize import minimize


class Contract(object):
    """ Base Class for defining contracts  """
    def __init__(self,beta,y=None):         # constructor method to set default params.
        self.beta  = beta                   # present bias in β-δ framework
        self.rho   = 0.95                   # 1/rho = elasticity of substitution
        if y is None:
            self.y = np.array([100,100,100])
        else:
            self.y = y
        self.r     = 0.0                     # bank's opportunity cost of funds
        self.delta = 1/(1+self.r)            # agent's psychic discount factor

    def print_params(self):
        """ print out parameters """
        params = vars(self)
        for p in sorted(params):    # print attributes alphabetically
            print("{0:<7} : {1}".format(p,params[p]))

    def u(self,ct):
        """ utility function """
        return  ( (1/(1-self.rho)) * ct**(1-self.rho)  )

    def PV(self,c):
        """discounted present value of any stream c"""
        return c[0] + np.sum(c[1:])

    def PVU(self,c, beta):
        """discounted present utility value of any stream c"""  
        return  self.u(c[0]) + beta * sum([self.u(ct) for ct in c[1:]] )

    def profit(self,c,y):
        """ present value of lender profits when exchanges c for y"""
        return  self.PV(y)-self.PV(c)

    def indif(self, ubar, beta):
        """ returns u(c1, c2) for graphing indifference curves in c1-c2 space.
        if beta = 1, will describe self 0's preferences """
        def idc(c1):
            return np.array((((1-self.rho)/(beta*self.delta))
                  *(ubar-self.u(c1)))**(1/(1-self.rho)))
        return idc

    def isoprofit(self, prfbar, y):
        """ returns profit(c1, c2) for graphing isoprofit lines in c1-c2 space.
        """
        def isoprf(c1):
            """isoprofit function isoprf(c1) """
            return np.array(y[1] + y[2] - prfbar) - c1
        return isoprf


class Monopoly(Contract):                    # build on contract class
    """ Class for solving Monopoly equilibrium contracts  """
    def __init__(self,beta):
        super(Monopoly,self).__init__(beta)    # inherit parent class properties
        self.kappa  = 0                        # cost of renegotiation
        self.guess  = self.y                   # initial guess for solver

    def fcommit(self):
        """monopolist optimal full commitment contractwith period0 self
        from closed form solution for CRRA"""
        A = ((self.PVU(self.y,self.beta) * (1-self.rho) )**(1/(1-self.rho)) )
        B = (1 + self.beta**(1/self.rho) * \
            (self.delta + self.delta**2))**(1/(self.rho-1))
        c0 = A*B
        c1 = c0 *(self.beta * self.delta *(1+self.r)  )**(1/self.rho)
        c2 = c0 *(self.beta * self.delta**2 *(1+self.r)**2  )**(1/self.rho)
        return np.array([c0,c1,c2])

    def negprofit(self,c):
        """ Negative profits (for minimization)"""
        return  -(self.PV(self.y) - self.PV(c))

    def reneg(self,c):
        """ Renegotiated contract offered to period-1-self
        c_0 is past but (c_1,c_2) now replaced by (cr_1, cr_2)"""
        PU =  self.u(c[1]) + self.beta*self.delta*self.u(c[2])
        A  =  (PU *(1-self.rho) )**(1/(1-self.rho))
        B = (1 + self.beta**(1/self.rho) * self.delta)**(1/(self.rho-1))
        cr0 = c[0]
        cr1 = A*B
        cr2 = cr1 *(self.beta * self.delta *(1+self.r)  )**(1/self.rho)
        return np.array([cr0,cr1,cr2])

    def reneg_proof_cons(self,c):
        """ the renegotiation-proof constraint gain from renegotiation
        cannot exceed its cost kappa"""
        return  -(self.profit(self.reneg(c),self.y)
                  -  self.profit(c,self.y) - self.kappa)

    def participation_cons(self,c):
        return (self.PVU(c,self.beta)  - self.PVU(self.y,self.beta))

    def reneg_proof(self):
        """calculate renegotiation-proof contract
        supplies constraints to solver that bank can't profit too much
        and period 0 borrower participation"""

        cons = ({'type': 'ineq',
                 'fun' : self.reneg_proof_cons },
                {'type': 'ineq',
                 'fun' : self.participation_cons })
        res = minimize(self.negprofit, self.guess, method='COBYLA',
                     constraints = cons)
        return res

class Competitive(Contract):                    # build on contract class
    """ Class for solving competitive equilibrium contracts  """
    def __init__(self, beta):
        super(Competitive,self).__init__(beta)  # inherits parent class properties
        self.kappa  = 0                         # cost of renegotiation
        self.guess  = self.y                    # initial guess for solver

    def fcommit(self):
        """competitive optimal full commitment contractwith period0 self
        from closed form solution for CRRA"""
        B = self.beta**(1/self.rho)
        D = self.PV(self.y)
        C = (1 + 2*B)
        c0 = D/C
        c1 = B * c0
        c2 = B * c0
        return np.array([c0,c1,c2])

    def negPVU(self,c):
        """0 self negative present utility value of 
        any stream c for minimization call"""
        return  - self.PVU(c, self.beta)

    def renegC(self, c):
        """ Renegotiated contract offered to period-1-self
        c_0 is past but (c_1,c_2) now replaced by (cr_1, cr_2)"""
        PV =  c[1] + c[2] - self.kappa
        B  =  self.beta**(1/self.rho)
        cr0 = c[0]
        cr1 = PV/(1+B)
        cr2 = B*cr1
        return np.array([c[0],cr1,cr2])
        
    def reneg(self,c):
        """ Renegotiated contract offered to period-1-self
        c_0 is past but (c_1,c_2) now replaced by (cr_1, cr_2)"""
        PU =  self.u(c[1]) + self.beta*self.delta*self.u(c[2])
        A  =  (PU *(1-self.rho) )**(1/(1-self.rho))
        B = (1 + self.beta**(1/self.rho) * self.delta)**(1/(self.rho-1))
        cr0 = c[0]
        cr1 = A*B
        cr2 = cr1 *(self.beta * self.delta *(1+self.r)  )**(1/self.rho)
        return np.array([cr0,cr1,cr2])
        
    def reneg_proof_cons(self,c):
        """ the renegotiation-proof constraint gain from renegotiation
        cannot exceed its cost kappa"""
        return  -(self.profit(self.reneg(c),self.y)
                  -  self.profit(c,self.y) - self.kappa)
                  
    def reneg_proof_consC(self,c):
        """ renegotiation-proof constraint gain from renegotiation
        goes to customer """
        cr = self.renegC(c)[1:]   #last two periods
        return  -(self.PVU(cr,self.beta)
                  -  self.PVU(c[1:],self.beta))


    def participation_cons(self,c):
        return (self.PV(self.y) - self.PV(c))

    def reneg_proof(self, monop_reg = True):
        """calculate renegotiation-proof contract that maxes 0-self's utility.
        supplies constraints to solver that bank can't profit too much
        and period 0 borrower participation"""
        if monop_reg:
            cons = ({'type': 'ineq',
                 'fun' : self.reneg_proof_cons },
                {'type': 'ineq',
                 'fun' : self.participation_cons })
        else:
            print('reneg surplus to customer -- sensitive solns ')
            cons = ({'type': 'ineq',
                 'fun' : self.reneg_proof_consC },
                {'type': 'ineq',
                 'fun' : self.participation_cons })
        res=minimize(self.negPVU, self.guess, method='COBYLA',
                     constraints = cons)
                     
        return res

if __name__ == "__main__":
    
    print("Base contract")
    c = Contract(beta = 0.6)
    c.y = [60, 120, 120]
    c.print_params()

    print("Monopoly contract")
    
    cM = Monopoly(beta = 0.5)
    cM.y = [80, 110, 110]
    cM.rho = 0.95
    cM.print_params()

    print("Competitive contract")
    cC = Competitive(beta = 0.5)
    cC.rho = 0.95
    cC.y = [80, 110, 110]
    cC.print_params()
    
    

    cCF = cC.fcommit()
    cCr = cC.reneg(cCF)
    cC.guess = cCr
    cCRP = cC.reneg_proof().x
    print(cCRP)
     

    cMF = cM.fcommit()
    cMr = cM.reneg(cCF)
    cM.guess = cMr
    cMRP = cM.reneg_proof().x
    
   # Analytic closed forms competitive
  #  A = cC.beta ** (1/cC.rho)
   # cA0 = (sum(cC.y) - cC.kappa)/(1+2*cA)
   # cCRPa = np.array([cA0, ])

    def ccrpa(C):
        B = C.beta**(1/C.rho)
        D = 1/(1+(1+B)*((C.beta+B)/(1+B))**(1/C.rho))
        print("D is equal to",D)
        c0 = sum(C.y)*D
        c1 = (sum(C.y)-c0)/(1+B)
        c2 = B* c1
        return np.array([c0, c1, c2])
    
    print("testing cCRP")
    print(cCRP.sum())
    print("reneg(cCRP):",cC.reneg(cCRP))
    print("PVU(cCRP) :",cC.PVU(cCRP,cC.beta))
    
    cCRPa = ccrpa(cC)
    print("PVU(cCRPa) :",cC.PVU(cCRPa,cC.beta))
    print("PVU(cCF) :",cC.PVU(cCF,cC.beta))
    