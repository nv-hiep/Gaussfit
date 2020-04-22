import os
import pickle

import numpy             as np
import matplotlib.pyplot as plt

# modules
import fit
import optimizer
import parallel




## A class for Gaussian decomposition using Savitzky-Golay filter ##
 #
 # params 
 #
 # return 
 #
 # version 11/2019
 # author Nguyen Hiep ##
class decomposer(object):
    def __init__(self):
        self.par = {
            'SG_winlen'         : None,
            'SG_order'          : None,
            'training_results'  : None,
            'y_thresh'          : 5.,
            'derv2_thresh'      : 5.,
            'band_frac'         : 0.1, # or 10%
        }

    


    ## Load training data from file ##
     #
     # params 
     #
     # return 
     #
     # version 11/2019
     # author Nguyen Hiep ##
    def load_training_data(self, file):
        self.par['training_data'] = pickle.load(open(file, 'rb'))


    
    ## Training data ##
     #
     # params 
     #
     # return 
     #
     # version 11/2019
     # author Nguyen Hiep ##    
    def train(
        self,
        SG_winlen_init = None,
        SG_order_init  = None,
        learning_rate  = 0.9,
        epsilon        = 0.25, # step of SG_winlen and SG_order
    ):
        ''' Train data to solve for optimal values of SG_winlen (and SG_order) '''

        if ( (not SG_winlen_init) or (not SG_order_init) ):
            print('Please choose initial guesses.')
            return
        
        if (not self.par['training_data']):
            print('Please load training data.')
            return
        
        print('Training...')

        self.par['SG_winlen'], self.par['SG_order'], self.par['training_results'] = optimizer.train(
                                        SG_winlen_init = SG_winlen_init,
                                        SG_order_init  = SG_order_init,
                                        training_data  = self.par['training_data'],
                                        y_thresh       = self.par['y_thresh'],
                                        derv2_thresh   = self.par['derv2_thresh'],
                                        epsilon        = epsilon,
                                        learning_rate  = learning_rate
                                    )


        print('Best parameters')

        print( 'Savitzky-Golay window size', self.par['SG_winlen'] )
        print( 'Savitzky-Golay order of polymonial', self.par['SG_order'] )






    ## Prepare to fit the data ##
     #
     # params 
     #
     # return 
     #
     # version 11/2019
     # author Nguyen Hiep ##    
    def to_fit(self, x, y, erry):
        ''' Decompose a single spectrum using current parameters '''

        if ( (not self.par['SG_winlen']) or (not self.par['SG_order']) ):
            print('SG_winlen or SG_order is unset')
            return

        status, results = fit.fit(
            x,
            y,
            erry,
            SG_winlen    = self.par['SG_winlen'],
            SG_order     = self.par['SG_order'],
            y_thresh     = self.par['y_thresh'],
            band_frac    = self.par['band_frac'],
            derv2_thresh = self.par['derv2_thresh']
        )
        return results


    

    ## Set param ##
     #
     # params 
     #
     # return 
     #
     # version 11/2019
     # author Nguyen Hiep ##    
    def set(self, key, value):
        if key in self.par:
            self.par[key] = value
        else:
            print('Given key does not exist.')


    

    ## Prepare to fit the data using parallel computation ##
     #
     # params 
     #
     # return 
     #
     # version 11/2019
     # author Nguyen Hiep ##    
    def parallel_fit(self, datfile):

        # Save some data to file to allow multiprocessing
        pickle.dump(
            [self, datfile], open('data/xtemp.pickle', 'wb')
        )

        
        # Parallel computation
        import parallel

        parallel.init()
        reslist = parallel.func()
        
        print('OK !')

        ret_keys = [
                    'idx',
                    'hgts',
                    'fwhms',
                    'cens',
                    'idx_init',
                    'hgts_init',
                    'fwhms_init',
                    'cens_init',
                    'sighgts',
                    'sigfwhms',
                    'sigcens',
                    'rchi2'
                   ]

        ret = dict((key, []) for key in ret_keys)

        for (i, res) in enumerate(reslist):

            # Save best-fit parameters
            ngauss = res['N_gauss']
            hgts   = res['fit_pars'][0:ngauss] if ngauss > 0 else []
            fwhms  = ( res['fit_pars'][ngauss: 2 * ngauss] if ngauss > 0 else []  )
            cens   = ( res['fit_pars'][2 * ngauss: 3 * ngauss] if ngauss > 0 else [] )

            ret['hgts'].append(hgts)
            ret['fwhms'].append(fwhms)
            ret['cens'].append(cens)
            ret['idx'].append([i for j in range(ngauss)])

            # Save initial guesses
            ngauss_init = len(res['init_pars']) // 3
            hgts_init   = ( res['init_pars'][0:ngauss_init] if ngauss_init > 0 else [] )
            fwhms_init  = ( res['init_pars'][ngauss_init: 2 * ngauss_init] if ngauss_init > 0 else [] )
            cens_init   = ( res['init_pars'][2 * ngauss_init: 3 * ngauss_init] if ngauss_init > 0 else [] )

            ret['cens_init'].append(cens_init)
            ret['fwhms_init'].append(fwhms_init)
            ret['hgts_init'].append(hgts_init)
            ret['idx_init'].append([i for j in range(ngauss_init)])

            # Fit errors
            rchi2    = [res['rchi2']] if 'rchi2' in res else None
            sighgts  = res['fit_err'][0:ngauss] if ngauss_init > 0 else []
            sigfwhms = ( res['fit_err'][ngauss: 2 * ngauss] if ngauss_init > 0 else [] )
            sigcens  = ( res['fit_err'][2 * ngauss: 3 * ngauss] if ngauss_init > 0 else [] )

            ret['rchi2'].append(rchi2)
            ret['sigcens'].append(sigcens)
            ret['sigfwhms'].append(sigfwhms)
            ret['sighgts'].append(sighgts)
        # End - for (reslist)

        print('100 finished.%')
        return ret











    ## Plot results ##
     #
     # params 
     #
     # return 
     #
     # version 11/2019
     # author Nguyen Hiep ##    
    def _plot(
        self,
        spec,
        data,
        idx,
        xlabel  = 'x [channels]',
        ylabel  = 'T [K]',
        xlim    = None,
        ylim    = None,
        guesses = False):

        # Extract info from data (must contain 'fit' categories)
        x = spec['x'][0]
        y = spec['y'][0]

        fwhms = data['fwhms'][idx]
        hgts  = data['hgts'][idx]
        cens  = data['cens'][idx]
    
        fwhms_init = data['fwhms_init'][idx]
        hgts_init  = data['hgts_init'][idx]
        cens_init  = data['cens_init'][idx]
    
        ngauss = len(hgts)

        plt.figure(figsize=(10,10))
    
        # if True:
        #     fwhms_true = data['fwhms'][idx]
        #     hgts_true  = data['hgts'][idx]
        #     cens_true  = data['cens'][idx]
    
        plt.plot(x, y, '-k', label='data', lw=1.5)
    
        # Plot fitted
        sum_fit = x * 0.
        plt.plot(0., 0., '-b', lw=1.0, label='fit cpnts')
        plt.plot(0., 0., '--k', lw=1, label='Guesses')
        for (hgt, fwhm, cen, hgt_init, fwhm_init, cen_init) in zip(hgts, fwhms, cens, hgts_init, fwhms_init, cens_init):
            yfit       = hgt * np.exp(-(x - cen) ** 2 / 2. / (fwhm / 2.355) ** 2)
            y_guess = hgt_init * np.exp( -(x - cen_init) ** 2 / 2. / (fwhm_init / 2.355) ** 2 )
            sum_fit     = sum_fit + yfit
            plt.plot(x, yfit, '-b', lw=1.0)
            
            if (guesses):
                plt.plot(x, y_guess, '--k', lw=1)
            
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)

            if (xlim):
                plt.xlim(*xlim)

            if (ylim):
                plt.ylim(*ylim)
        # End - for
        
        plt.plot(x, sum_fit, '-r', lw=1.5, label='Fit')
    
        # Plot True components
        # sum_true = x * 0.
        # if True:
        #     for (hgt, fwhm, cen) in zip(hgts_true, fwhms_true, cens_true):
        #         y_true   = hgt_true * np.exp(-(x - cen_true) ** 2 / 2.0 / (fwhm_true / 2.355) ** 2)
        #         sum_true = sum_true + y_true
        #         plt.plot(x, y_true, '-r', lw=0.5)
        #     # End - for

        #     plt.plot(x, sum_true, '-r', lw=1.0, label='True')
        # End - if
    
        plt.title('index = {0}, ngauss = {1}'.format(idx, ngauss), fontsize=16)
        
        plt.legend(loc=0)
        plt.show()