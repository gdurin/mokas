"""
fitting code, ready to be used by mokas
"""
import sys
from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
# define a number of fitting functions



class Size_Distribution:
    def __init__(self, n_params=3):

        if n_params == 3:
            self.theory = self._th_PS_3p
        elif n_params == 4:
            self.theory = self._th_PS_4p

        p = ['A', 'tau', 'S0', 'n']
        self.params = p[:n_params]
        self.n_params = n_params

    def _th_PS_3p(self, p, S):
        return p[0]*S**(-p[1])*np.exp(-(S/p[2]))

    def _th_PS_4p(self, p, S):
        return p[0]*S**(-p[1])*np.exp(-(S/p[2])**p[3])

    @property
    def repr(self):
        if self.n_params == 3:
            return r"$A S^{-\tau} exp(-S/S_0)$"
        elif self.n_params == 4:
            return r"$A S^{-\tau} exp(-(S/S_0)^n)$"

class LogSize_Distribution:
    def __init__(self, n_params=3):

        if n_params == 3:
            self.theory = self._th_PS_3p
        elif n_params == 4:
            self.theory = self._th_PS_4p

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
    def __init__(self, x, y, theory, p0, linlog='log'):
        self.x = x
        self.y = y
        self.p0 = p0
        self.theory = theory
        self.linlog = linlog

    def residual(self, _params):
        if self.linlog == 'lin':
            return self.theory(_params, self.x) - self.y
        else:
            return np.log10(self.theory(_params, self.x)) - np.log10(self.y)

    def get_params(self):
        full_output = leastsq(self.residual, self.p0, full_output=True)
        params, covmatrix, infodict, mesg, ier = full_output
        print(mesg)
        params_err = self._get_params_err(params, covmatrix)
        return params, params_err

    def _get_params_err(self, params, covmatrix):
        res = self.residual(params)
        cst = np.dot(res, res)
        lenData = len(self.x)
        print("Cost: %f" % cst)
        # Standard error of the regression
        ser = (cst/(lenData - len(params)))**0.5
        stDevParams = [covmatrix[i,i]**0.5 * ser for i in range(len(params))]
        return stDevParams

if __name__ == "__main__":
    plt.close("all")
    S = np.array([   5.61009227,    7.06268772,    8.89139705,   11.19360569,
         14.09191466,   17.74066946,   22.33417961,   28.11706626,
         35.39728922,   44.56254691,   56.10092272,   70.62687723,
         88.9139705 ,  111.93605693,  140.91914656,  177.40669462,
        223.34179608,  281.1706626 ,  353.97289219,  445.62546907,
        561.00922715,  706.26877231,  889.13970502, 1119.36056928,
       1409.19146563, 1774.06694617])
    PS = np.array([1.85010628e-01, 1.14119129e-01, 1.60101405e-01, 1.38051414e-01,
       9.11417531e-02, 7.09256592e-02, 6.52952572e-02, 3.90798476e-02,
       3.39089227e-02, 2.69998747e-02, 2.14467628e-02, 1.65021186e-02,
       1.17385959e-02, 9.29839733e-03, 6.08983271e-03, 4.00386786e-03,
       2.62219522e-03, 1.66012018e-03, 1.04020125e-03, 5.20479512e-04,
       2.68730522e-04, 9.85201112e-05, 5.54322584e-05, 1.29504141e-05,
       4.11475183e-06, 1.63423178e-06])
    S, PS = S[2:], PS[2:]
    n_params = 3
    p00 = [1, 1.1, 300, 1]
    p0 = p00[:n_params]
    log_data = True
    if log_data:
        plt.figure()
        sd = LogSize_Distribution(n_params)
        model = Model(np.log10(S), np.log10(PS), sd.theory, p0, 'lin')
        params, errors = model.get_params()
        print(sd.repr)
        for q in zip(sd.params, params, errors):
            print("%s: %.3f +/- %.3f" % q)
        plt.plot(np.log10(S), np.log10(PS), 'o')
        plt.plot(np.log10(S), sd.theory(params, np.log10(S)), '--', label=sd.repr)

    if True:
        plt.figure()
        n_params = 3
        p0 = p00[:n_params]
        sd = Size_Distribution(n_params)
        model = Model(S, PS, sd.theory, p0, 'log')
        params, errors = model.get_params()
        print(sd.repr)
        for q in zip(sd.params, params, errors):
            print("%s: %.2f +/- %.2f" % q)
        plt.loglog(S, PS, 'o')
        plt.loglog(S, sd.theory(params, S), '--', label=sd.repr)
    plt.legend()
    plt.show()