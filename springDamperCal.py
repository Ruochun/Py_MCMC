import numpy as np
import mckDataGen as DG

# import required packages
import scipy.io as sio
from pymcmcstat.MCMC import MCMC
from pymcmcstat.settings.DataStructure import DataStructure
import matplotlib.pyplot as plt
from time import time as timetest
import ctypes
from numpy.ctypeslib import ndpointer

def main():
    num_total_data = 10000
    num_train = 300
    num_test = num_total_data - num_train

    # TrainData format is (m, c, k, r, u, F, a)
    TrainData = DG.mckDataGen(num_total_data)
    # TrainData = DG.mckDataGen(num_total_data, mB=0, cB=0, kB=0)

    # assign initial values
    theta0 = {'m': 1., 'c': 1., 'k': 1.}
    # It's stupid; due to the implementation, ssfun args must be np arrays, not Python dicts
    theta0vec = list(theta0.values())

    mcstat = MCMC()
    mcstat.data.add_data_set(x = TrainData[:num_train,:-1],
                             y = TrainData[:num_train,-1].flatten())
    mcstat.parameters.add_model_parameter(name='$m$', theta0=theta0['m'], minimum=0, sample=True)
    mcstat.parameters.add_model_parameter(name='$c$', theta0=theta0['c'], minimum=0, sample=True)
    mcstat.parameters.add_model_parameter(name='$k$', theta0=theta0['k'], minimum=0, sample=True)

    n = 10
    st = timetest()
    for ii in range(n):
        sstest = ssfun(theta0vec, mcstat.data)
    et = timetest()
    print('SOS function evaluation time: {} ms'.format((et - st)/n*1e3))
    mcstat.model_settings.define_model_settings(sos_function = ssfun)
    mcstat.simulation_options.define_simulation_options(nsimu = 5e4,
                                                        updatesigma=True)

    # Run MCMC
    mcstat.run_simulation()
    
    # Inspect results
    results = mcstat.simulation_results.results
    names = results['names']
    fullchain = results['chain']
    fulls2chain = results['s2chain']
    nsimu = results['nsimu']
    burnin = int(nsimu/2)
    chain = fullchain[burnin:, :]
    s2chain = fulls2chain[burnin:, :]

    mcstat.chainstats(chain, results)

    # plot chain metrics
    p1 = mcstat.mcmcplot.plot_chain_panel(chain, names, figsizeinches=(4, 4));
    mcstat.mcmcplot.plot_density_panel(chain, names, figsizeinches=(4, 4));
    mcstat.mcmcplot.plot_pairwise_correlation_panel(chain, names,
						    figsizeinches=(4, 4));
    p1.savefig('p1.png')

def ssfun(theta, data):
    # m, c, k are calibration params
    theta_dict = {'m': theta[0], 'c': theta[1], 'k': theta[2]}
    m = theta_dict['m']
    c = theta_dict['c']
    k = theta_dict['k']
    # r, u, F are sim params that we do not calibrate
    r = data.xdata[0][:,3]
    u = data.xdata[0][:,4]
    F = data.xdata[0][:,5]
    # define spring--damper model
    s_model = DG.evalModel(m, c, k, r, u, F)
    # Residual
    res = data.ydata[0][:].flatten() - s_model
    ss = np.dot(res.T, res)
    return ss

if __name__ == "__main__":
    main()


