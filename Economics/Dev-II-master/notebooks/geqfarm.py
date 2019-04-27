# -*- coding: utf-8 -*-
# farmeconomy.py   module
# authors: Jonathan Conning & Aleks Michuda
# An OOP implementation

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from collections import namedtuple


class Economy(object):
    """ Economy with an Equilibrium Farm Size Distribution
   At present the initial distribution of skills is uniformly distributed.
   For example N = 5 and s = np.array([1, 1, 1, 1, 1.5]) has 5 farmer groups.
   We take the landlord class to be last indexed group .
    """
    def __init__(self, N):  # constructor to set initial parameters.
        self.N       = N   # of quantiles (number of skill groups)
        self.GAMMA   = 0.8    # homogeneity factor
        self.ALPHA   = 0.5    # alpha (land) for production function
        self.LAMBDA  = 1.0/N    # landlord share of labor
        self.TBAR    = 100    # Total Land Endowment
        self.LBAR    = 100    # Total Labor Endowment
        self.H       = 0.0    # fixed cost of production
        self.s       = np.ones(N)

    def prodn(self, X, s):
        Y = s*((X[0]**self.ALPHA)*(X[1]**(1-self.ALPHA)))**self.GAMMA
        return Y

    def marginal_product(self, X, s):
        """ Production function technoogy """
        MPT = self.ALPHA*self.GAMMA*self.prodn(X,  s)/X[0]
        MPL = (1-self.ALPHA)*self.GAMMA*self.prodn(X, s)/X[1]
        return np.append(MPT, MPL)

    def profits(self, X, s, w):
        """ profits given factor prices and (T, L, s)"""
        return self.prodn(X, s) - np.dot(w, X) - self.H

    def demands(self, w, s):
        """Returns competitive demands for each skill group in a subeconomy
        with factor supply (tbar, lbar) and vector s.
        """
        alpha, gamma = self.ALPHA, self.GAMMA
        land = ((w[1]/(gamma*s*(1-alpha))) *
                (((1-alpha)/alpha)*(w[0]/w[1])) **
                (1-gamma*(1-alpha)))**(1/(gamma-1))
        labor = ((w[0]/(gamma*s*alpha)) *
                 ((alpha/(1-alpha))*(w[1]/w[0])) **
                 (1-gamma*alpha))**(1/(gamma-1))
        # zero demands if fixed cost implies negative profits (numeric only)
        X = np.array([land, labor])
        profitable = (self.profits(X, s, w) > 0)
        return X*profitable

    def excessD(self, w, Xbar, s):
        """ Total excess land and labor demand given factor prices in
        subeconomy with Xbar supplies
        returns excess demand in each market
        """
        res = np.array(([np.sum(self.demands(w, s)[0])-Xbar[0],
                         np.sum(self.demands(w, s)[1])-Xbar[1]]))
        return res

    def smallhold_eq(self, Xbar, s, analytic=True):
        """ Solves for market clearing factor prices (analytically or
        numerically) for subeconomy given Xbar supplies
        When numerically solved: minimizes sum of squared excess demands
        Returns a named tuple with factor prices and demands c_eqn.w, c_eqn.D
        """
        if analytic:     # for specific CobbDouglas
            gamma = self.GAMMA
            s_fringe, s_R = s[0:-1], s[-1]
            psi = np.sum((s_fringe/s_R)**(1/(1-gamma)))
            Lr = Xbar[1]/(1+psi)
            Tr = Xbar[0]/(1+psi)
            L_fringe = Lr*(s_fringe/s_R)**(1/(1-gamma))
            T_fringe = Tr*(s_fringe/s_R)**(1/(1-gamma))
            Xs = np.array([np.append(T_fringe, Tr), np.append(L_fringe, Lr)])
            WR = self.marginal_product(Xs[:, -2], s[-2])
        else:  # Numeric solution should work for any demands
            w0 = np.array([0.5, 0.5])
            f = lambda w: np.sum(self.excessD(w, Xbar, s)**2)
            res = minimize(f, w0, method='Nelder-Mead')
            WR = res.x
        Xs = self.demands(WR, s)
        result = namedtuple('result', ['w', 'X'])
        res = result(w=WR, X=Xs)
        return res

    def cartel_income(self, Xr, theta):
        """ Cartel group's income from profits and factor income

        when cartel uses (tr,lr) fringe has (TBAR-tr,LBAR-lr)  """
        # at present cartel is always last index farm
        s_fringe, s_R = self.s[0:-1], self.s[-1]  # landlord is top index farmer
        TB_fringe = max(self.TBAR - Xr[0], 0)
        LB_fringe = max(self.LBAR - Xr[1], 0)
        fringe = self.smallhold_eq([TB_fringe, LB_fringe], s_fringe)
        y = self.prodn(Xr, s_R) - \
            np.dot(fringe.w, [Xr[0]-self.TBAR*theta,
                              Xr[1]-self.LAMBDA*self.LBAR])
        # print("cartel:  Tr={0:8.3f}, Lr={1:8.3f}, y={2:8.3f}".format(Xr[0],
        # Xr[1],y))
        return y

    def cartel_eq(self, theta, guess=[1, 1]):
        """ Cartel chooses own factor use (and by extension how much to
        withold from the fring to max profits plus net factor sales)
        """
        f = lambda X: -self.cartel_income(X, theta)
        res = minimize(f, guess, method='Nelder-Mead')
        XR = res.x
        # print('XR:',XR)
        fringe = self.smallhold_eq([self.TBAR, self.LBAR]-XR, self.s[0:-1])
        XD = np.vstack((fringe.X.T, XR)).T
        WR = fringe.w

        result = namedtuple('result', ['w', 'X'])
        cartel_res = result(w= WR, X= XD)
        return cartel_res

    def print_params(self):
        """ print out parameters """
        params = vars(self)
        count = 1
        for p in sorted(params):    # print attributes alphabetically
            if (count % 4) == 0:
                print("{0:<6} : {1}".format(p,params[p]))
            else:
                print("{0:<6} : {1}".format(p,params[p])),
            #print(count)
            count += 1
            
    def fooprint(self):
        print(self.print_params())

class MirEconomy(Economy):
    """ sub class of Economy class but with Mir as subeconomy
    """
    def __init__(self, N):  # constructor to set initial parameters.
        super(MirEconomy, self).__init__(N)  # inherit properties
        # if None supplied use defaults



class CESEconomy(Economy):
    """ sub class of Economy class but with two factor CES
    """
    def __init__(self, N):  # constructor to set initial parameters.
        super(CESEconomy, self).__init__(N)  # inherit properties
        # if None supplied use defaults
        self.N         = N # of quantiles (number of skill groups)
        self.RHO       = 0.8    # homogeneity factor
        self.PHI       = 0.5    # alpha (land) for production function
        self.aL        = 1.0    # landlord share of labor
        self.aT        = 1.1    # Total Land Endowment

    def prodn(self, X, s):
        Y = s*(self.PHI*X[0]**(self.RHO) + (1-self.PHI)*X[1]**(self.RHO))  \
            ** (self.GAMMA/self.RHO)
        return Y

    def marginal_product(self, X, s):
        """ Production function technoogy """
        common = s*(self.PHI*X[0]**self.RHO+(1-self.PHI)*X[1]**self.RHO) \
            ** ((1+self.RHO)/self.RHO)
        MPT = common * self.PHI*X[0]**(-self.RHO-1)
        MPL = common * (1-self.PHI)*X[1]**(-self.RHO-1)
        return np.append(MPT, MPL)


#########################
# We also define some utility functions

# Note these print statements format correctly in python2.7 not 3.4
def scene_print(ECO, numS=5,prnt=True,detail=True):
        """Creates numS land ownership (theta) scenarios
        and returns competitive and market-power distorted equilibria  
        Prints results if flags are on.
        
        Args:
          ECO -- Instance of an Economy object 
          numS -- number of values of theta
          prnt -- print table if True
        Returns:
          [Xc,Xr,wc,wr]  where
            Xrc -- Efficient/Competitive landlord factor use
            Xr -- numS x 2 matrix, Xr[theta] = Landlords' distorted use
            wc -- competitive factor prices
            wr -- wr[theta] distorted competitive factor prices
        
        """
        import sys
        if not sys.version_info[0] ==2:
            print('Use python 2.x for table formats')
        print("Running {0} scenarios...".format(numS))                
        # competitive eqn when landlord is just part of the competitive fringe        
        comp = ECO.smallhold_eq([ECO.TBAR,ECO.LBAR],ECO.s)
        wc, Xc = comp.w, comp.X         
        Xrc = Xc[:,-1]   # landlord's factor use
        
        #
        guess = Xrc    
        # distorted equilibria at different land ownership theta
        theta = np.linspace(0,1,numS+1)
        theta[-1] = 0.99
        if prnt:        
            print("\nAssumed Parameters")
            print("==================")
            ECO.print_params()  
            print('\nEffcient:[ Trc, Lrc]      [rc,wc]       w/r   '),
            if detail:
                print('F( )    [r*Tr]  [w*Lr]'),
            print("")    
            print("="*78)
            print("        [{0:6.2f},{1:6.2f}] ".format(Xrc[0],Xrc[1])),
            print("[{0:4.2f},{1:4.2f}]".format(wc[0],wc[1])),
            print("  {0:4.2f} ".format(wc[1]/wc[0])),  
            if detail:
                print("| {0:5.2f} ".format(ECO.prodn(Xrc,ECO.s[-1]))),
                print(" {0:5.2f} ".format(Xrc[0]*wc[0])),
                print(" {0:6.2f} ".format(Xrc[1]*wc[1]))
        
            print("\nTheta  [ Tr, Lr ]      [rM,wM]        w/r  |"),
            print('F()   [T_hire]  [T_sale] [L_hire]')
            
            print("="*78)

        Xr = np.zeros(shape=(numS+1,2))  # Xr - lord factor use for each theta
        wr = np.zeros(shape=(numS+1,2))            
        for i in range(numS+1):           
            cartelEQ = ECO.cartel_eq(theta[i], guess)
            Xr[i] = cartelEQ.X[:,-1]
            wr[i] = cartelEQ.w
            guess = Xr[i]
            if prnt:            
                print(" {0:3.2f}".format(theta[i])),
                print(" [{0:6.2f},{1:6.2f}]".format(Xr[i,0],Xr[i,1])),
                print("[{0:5.2g},{1:5.2f}] {2:5.2f}" \
                .format(wr[i,0],wr[i,1],wr[i,1]/wr[i,0])),
                if detail:
                    print("| {0:5.2f} ".format(ECO.prodn(Xr[i],ECO.s[-1]))),      
                    print(" {0:6.2f} ".format(Xr[i,0]*wr[i,0])),
                    print(" {0:6.2f} ".format(theta[i]*ECO.TBAR*wr[i,0])),
                    print(" {0:6.2f} ".format(Xr[i,1]*wr[i,1])),
                print("")
        if prnt:
            print("="*78)
            
        return (Xrc, Xr, wc, wr)

def factor_plot(ECO, Xrc, Xr):
    plt.rcParams["figure.figsize"] = (10, 8)
    numS = len(Xr)-1
    theta = np.linspace(0, 1, numS+1)
    Tr, Lr = Xr[:, 0], Xr[:, 1]
    Tr_net = Tr-np.array(theta) * ECO.TBAR
    Lr_net = Lr - ECO.LAMBDA * ECO.LBAR
    # print(Tr_net, Lr_net)
    Trc_net = Xrc[0]*np.ones(numS+1)-np.array(theta)*ECO.TBAR
    Lrc_net = Xrc[1]*np.ones(numS+1)-ECO.LAMBDA*ECO.LBAR
    plt.grid()
    plt.plot(theta, Tr_net, '-ro', label='distorted land')
    plt.plot(theta, Trc_net, label='efficient land')
    plt.plot(theta, Lr_net, '-bx', label='distorted labor')
    plt.plot(theta, Lrc_net, label='efficient labor')
    plt.grid()
    plt.ylim(-100, ECO.TBAR)
    # plt.xlabel(r'$\gamma =$')
    plt.title('Landlord net factor hire for '+r'$\gamma =$ {0}'
              .format(ECO.GAMMA))
    plt.xlabel(r'$\theta$ -- Landlord land ownership share')
    plt.legend(loc='lower left',title='net hiring in of')
    plt.show()
    return

def TLratio_plot(ECO, Xrc, Xr):
    plt.rcParams["figure.figsize"] = (10, 8)
    numS = len(Xr)-1
    theta = np.linspace(0, 1, numS+1)
    plt.plot(theta, Xr.T[0][:]/Xr.T[1][:], '-ro', label='distorted')
    plt.plot(theta, (Xrc[0]/Xrc[1])*np.ones(numS+1), '--', label='efficient')
    plt.title('Land to labor ratio on landlord farm '+r'$\gamma =$ {0}'
              .format(ECO.GAMMA))
    plt.xlabel(r'$\theta$ -- Landlord land ownership share')
    plt.legend(loc='upper left',title='Land/Labor ratio')
    plt.show()
    return



#
if __name__ == "__main__":
    """Sample use of the Economy class """

    s = np.array([1.,  1.,  1.,  1.,  1.])
    N = len(s)
    E = Economy(N)    # an instance takes N length as parameter
    E.ALPHA = 0.5
    E.GAMMA = 0.90

    E.smallhold_eq([E.TBAR, E.LBAR], s, analytic=True)
    
    (Xrc, Xr, wc, wr) = scene_print(E, 10, detail=True)
    
    #factor_plot(E,Xrc,Xr)
    TLratio_plot(E,Xrc,Xr)

#    scene_plot(E, Xrc, Xr)
#    plt.show()
##
#    print("Competitive Fringe cases EE(C) and DE cases")
#    DE = Economy(5)
#    DE.smallhold_eq([E.TBAR, E.LBAR], s, analytic=True)
#    (Xrc, Xr, wc, wr) = scenarios(DE, 10, detail=True)
#    scene_plot(E, Xrc, Xr)
#
#    print("MIR Fringe cases DD")
#    DD = MirEconomy(5)
#    print(DD.prodn(np.array([10, 10]), np.array([1, 1.05])))
#    DD.smallhold_eq([E.TBAR, E.LBAR], s, analytic=True)
#    (Xrc, Xr, wc, wr) = scenarios(DD, 10, detail=True)
#    scene_plot(E, Xrc, Xr)
##
##    print("TESTING CES")
##    CE = CESEconomy(5)
##    print CE.prodn(np.array([10, 10]), np.array([1, 1.05]))
##    CE.smallhold_eq([E.TBAR, E.LBAR], s, analytic=True)
##    (Xrc, Xr, wc, wr) = scenarios(CE, 10, detail=True)
##    scene_plot(E, Xrc, Xr)
