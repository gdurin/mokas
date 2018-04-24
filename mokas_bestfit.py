"""
fitting code, ready to be used by mokas
"""
import sys
import scipy
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
# define a number of fitting functions



class Size_Distribution:
    def __init__(self, n_params=3):

        if n_params == 3:
            self.y = self._th_PS_3p
        elif n_params == 4:
            self.y = self._th_PS_4p

        p = ['A', 'tau', 'S0', 'n']
        self.params = p[:n_params]
        self.n_params = n_params

    def _th_PS_3p(self, p, S):
        return p[0]*S**(-p[1])*np.exp(-(S/p[2]))

    def _th_PS_4p(self, p, S):
        return p[0]*S**(-p[1])*np.exp(-(S/p[2])**p[3])

    def jacobian(self, p, S, sigma=1.):
        if self.n_params == 3:
            y = self.y(p, S)
            dy_dA = y / p[0]
            dy_tau = - y * np.log(S)
            dy_S0 = S * y / p[2]**2
            jac = np.array([dy_dA, dy_tau, dy_S0])
            jac = jac / sigma
            return np.transpose(jac)
        elif self.n_params == 4:
            y = self._th_PS_4p(p, S)
            return None

    def p0_guess(self, S, PS):
        lenS = len(S)//2
        tau, logA = np.polyfit(np.log10(S[:lenS]), np.log10(PS[:lenS]),1)
        A = 10**(logA)
        S0, _A = np.polyfit(S[lenS:], np.log10(PS[lenS:]),1)
        if self.n_params == 3:
            p0 = [A, -tau, -1./S0]
            print("Initial guess: %.2f, %.2f, %.2f" % tuple(p0))
            return p0
        elif self.n_params == 4:
            return [A, -tau, -1./S0, 1.]

    @property
    def repr(self):
        if self.n_params == 3:
            return r"$A S^{-\tau} exp(-S/S_0)$"
        elif self.n_params == 4:
            return r"$A S^{-\tau} exp(-(S/S_0)^n)$"

class LogSize_Distribution:
    def __init__(self, n_params=3):

        if n_params == 3:
            self.y = self._th_PS_3p
        elif n_params == 4:
            self.y = self._th_PS_4p

        p = ['logA', 'tau', 'S0', 'n']
        self.params = p[:n_params]
        self.n_params = n_params

    def _th_PS_3p(self, p, x):
        return p[0] - p[1] * x - 10**(x)/p[2]*np.log10(np.e)

    @property
    def repr(self):
        if self.n_params == 3:
            return r"$logA - \tau log(S) - 10**S/S_0)$"
        elif self.n_params == 4:
            return r"$A S^{-\tau} exp(-(S/S_0)^n)$"


class Model():
    """
    link data to theory, provides residual and cost function
    """
    def __init__(self, x, y, theory, p0=None, y_err=None, linlog='log', use_jacobian=False):
        self.x = x
        self.y = y
        self.theory = theory
        self.linlog = linlog
        if y_err is not None:
            self.sigma = y_err
        else:
            self.sigma = 1.
        self.use_jacobian = use_jacobian
        if p0 is None:
            self.p0 = self.theory.p0_guess(x, y)
        self.maxfev = len(self.p0) * 1000


    def residual(self, _params):
        y = self.y
        P = self.theory.y(_params, self.x)
        if self.linlog == 'lin':
            return (P - y)/self.sigma
        elif self.linlog == 'log':
            #return np.log10(self.theory(_params, self.x)) - np.log10(self.y)
            return (scipy.log10(P/self.sigma) - scipy.log10(y/self.sigma))
            #return np.log10(y_th) - np.log10(y)

    def jacobian(self, _params):
        jac = self.theory.jacobian(_params, self.x, self.sigma)
        #print(jac)
        return jac
        
    def get_params(self):
        if self.use_jacobian:
            try:
                full_output = leastsq(self.residual, self.p0, 
                    col_deriv=self.jacobian, full_output=True, maxfev=self.maxfev)
            except:
                return [], [None], 0
        else:
            try:
                full_output = leastsq(self.residual, self.p0, full_output=True, maxfev=self.maxfev)
            except:
                return [], [None], 0

        params, covmatrix, infodict, mesg, ier = full_output
        print(mesg)
        print("%i iterations" % infodict['nfev'])
        params_err = self._get_params_err(params, covmatrix)
        if ier in range(1,5):
            return params, params_err, ier
        else:
            return params, len(params) * [None], ier

    def _get_params_err(self, params, covmatrix):
        res = self.residual(params)
        cst = np.dot(res, res)
        lenData = len(self.x)
        try:
            print("Cost: %f" % cst)
            # Standard error of the regression
            ser = (cst/(lenData - len(params)))
            stDevParams = [(covmatrix[i,i] * ser)**0.5 for i in range(len(params))]
            return stDevParams
        except:
            return None

if __name__ == "__main__":
    plt.close("all")
    S = np.array([  2.81838293,   4.46683592,   5.62341325,   7.07945784,
         8.91250938,  11.22018454,  14.12537545,  17.7827941 ,
        22.38721139,  28.18382931,  35.48133892,  44.66835922,
        56.23413252,  70.79457844,  89.12509381, 112.20184543,
       141.25375446, 177.827941  , 223.87211386, 281.83829313])


    PS = np.array([6.03918636e-02, 4.96146847e-02, 1.32969863e-02, 9.61434072e-03,
       1.16872056e-02, 9.75415774e-03, 5.17266614e-03, 4.05463973e-03,
       3.32478790e-03, 1.75146325e-03, 1.27128741e-03, 8.93676337e-04,
       4.79509103e-04, 2.42223463e-04, 1.27224278e-04, 4.98367412e-05,
       1.23158718e-05, 3.49387310e-06, 9.71348718e-07, 7.71569712e-07])

    PS_err =  np.array([9.80649538e-04, 5.64035195e-04, 2.36331203e-04, 1.59923820e-04,
       1.39911618e-04, 1.01628995e-04, 5.89226549e-05, 4.14615105e-05,
       2.98338999e-05, 1.72135753e-05, 1.16519032e-05, 7.76152769e-06,
       4.51695233e-06, 2.55039077e-06, 1.46827903e-06, 7.29987164e-07,
       2.88257843e-07, 1.21956242e-07, 5.10785550e-08, 3.61608513e-08])

    nmax = -8
    S, PS, PS_err = S[1:-1], PS[1:-1], PS_err[1:-1]
    n_params = 3
    p00 = [3, 1.1, 820, 1]
    p0 = p00[:n_params]
    # log_data = True
    # if not log_data:
    #     plt.figure()
    #     sd = LogSize_Distribution(n_params)
    #     #model = Model(S, PS, sd, p0, PS_err, 'log')
    #     model = Model(S, PS, sd, p0, PS_err, 'log')
    #     params, errors, ier = model.get_params()
    #     print(sd.repr)
    #     for q in zip(sd.params, params, errors):
    #         print("%s: %.3f +/- %.3f" % q)
    #     plt.plot(np.log10(S), np.log10(PS), 'o')
    #     plt.plot(np.log10(S), sd.theory(params, np.log10(S)), '--', label=sd.repr)

    if True:
        fig, ax = plt.subplots()
        #plt.figure()
        n_params = 3
        p0 = p00[:n_params]
        theory = Size_Distribution(n_params)
        #PS_err = None
        #model = Model(S, PS, theory, p0, PS_err,'log')
        model = Model(S, PS, theory, y_err=PS_err, p0=None, linlog='log', use_jacobian=False)
        params, errors, ier = model.get_params()
        print(theory.repr)
        for q in zip(theory.params, params, errors):
            print("%s: %.2f +/- %.2f" % q)
        #plt.loglog(S, PS, 'o')
        ax.loglog(S, PS, 'o')
        #plt.errorbar(S, PS, yerr=10*PS_err, ms=4, fmt="o", ecolor='g', capthick=2)
        ax.loglog(S, theory.y(params, S), '--', label=theory.repr)
        ax.legend()
    plt.show()