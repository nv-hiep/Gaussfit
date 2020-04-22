import time

import numpy as np
import copy  as cp
# import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from mpfit             import mpfit



## Take savitzky_golay of a fucntion: smooth data, and derivatives ##
 # from https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
 #
 # params 
 #
 # return 
 #
 # version -
 # author - ##
def savitzky_golay(y, window_size, order, deriv=0):
    r'''Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    
    This code has been taken from http://www.scipy.org/Cookbook/SavitzkyGolay
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.savefig('images/golay.png')
    #plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    '''
    msg = 'window_size and order have to be of type int'
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError (msg):
        raise ValueError('window_size and order have to be of type int')
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('window_size size must be a positive odd number')
    if window_size < order + 2:
        raise TypeError('window_size is too small for the polynomials order')
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')







## Gaussian function ##
 #
 # params 
 #
 # return 
 #
 # version 11/2017
 # author Nguyen Hiep ##
def gaussian(a, wid, x0):
    sigma = wid / 2.354820045  # (2 * sqrt( 2 * ln(2)))
    return lambda x: a * np.exp(-(x - x0) ** 2 / 2. / sigma ** 2)








## Multi-component Gaussian ##
 #
 # params 
 #
 # return 
 #
 # version 11/2017
 # author Nguyen Hiep ##
def func(x, *par):
    ''' Multi-component Gaussian

    Parameters= [amp1, ..., ampN, width1, ..., widthN, x01, ..., x0N],
    -> Length = 3 x Ng.
    '''
    ng = len(par) // 3
    y  = np.zeros(len(x))
    for i in range(ng):
        y = y + gaussian(par[i], par[i + ng], par[i + 2 * ng])(x)
    return y






## Function for MPFIT ##
 #
 # params 
 #
 # return 
 #
 # version 11/2017
 # author Nguyen Hiep ##
def func_mpfit(p, fjac=None, x=None, y=None, err=None):
    status = 0
    ng     = (len(p)) // 3

    hgt    = p[0    : int(ng*1)]
    cen    = p[int(ng*1) : int(ng*2)]
    wid    = p[int(ng*2) : int(ng*3)]

    # fcn = 0.
    # for hi,ci,wi in zip(hgt,cen, wid):
    #     fcn = fcn + hi*np.exp(- ( (x-ci)/(0.6005612*wi) )**2)

    fcn = np.zeros(len(x))
    for i in range(ng):
        fcn = fcn + gaussian(p[i], p[i + ng], p[i + 2 * ng])(x)

    return [status, (y-fcn)/err]




## Find Gaussian components from conditions of derivatives ##
 # Use Gausspy method and Savitzky-Golay filter
 #
 # params 
 #
 # return 
 #
 # version 11/2019
 # author Nguyen Hiep ##
def find_peaks(
    vel,
    ytb,
    errors       = None,
    SG_winlen    = None,
    SG_order     = None,
    y_thresh     = 5.,
    band_frac    = 0.1,
    derv2_thresh = 5.
):
    '''  Find initial guesses (Window length and Order of polynomial) using Savitzky-Golay filter

    ytb,              Input data
    dv,               x-spacing absolute units
    SG_winlen,        Window length for Savitzky-Golay filter
    SG_order          Order of polynomial for  Savitzky-Golay filter
    plot = False,     Show diagnostic plots?
    y_thresh = 5.0    Initial Spectrum S/N threshold
    band_frac         Edge fraction of data used for S/N threshold computation
    derv2_thresh      S/N threshold for Second derivative
    '''

    errors = None  # Until error

    if ( (not SG_winlen) or (not SG_order) ):
        print('Please choose value for Window length.')
        return

    if np.any(np.isnan(ytb)):
        print('NaN-values in data, cannot continue.')
        return

    # Data inspection
    vel  = np.array(vel)
    ytb  = np.array(ytb)
    dv   = np.abs(vel[1] - vel[0])
    fvel = interp1d(np.arange(len(vel)), vel)  # index -> x
    ylen = len(ytb)

    # Take derivatives
    SG_winlen = int( SG_winlen )
    SG_order  = int( SG_order )

    if ( SG_winlen % 2 != 1 ):
        SG_winlen = SG_winlen - 1

    y_smooth = savitzky_golay(ytb,      window_size=SG_winlen, order=SG_order, deriv=0)  # 45
    d1y      = savitzky_golay(y_smooth, window_size=SG_winlen, order=SG_order, deriv=1)
    d2y      = savitzky_golay(d1y,      window_size=SG_winlen, order=SG_order, deriv=1)
    d3y      = savitzky_golay(d2y,      window_size=SG_winlen, order=SG_order, deriv=1)
    d4y      = savitzky_golay(d3y,      window_size=SG_winlen, order=SG_order, deriv=1)


    # Data noise
    if (not errors):
        errors = np.std(ytb[0: int(band_frac * ylen)])
    # End-if

    thresh = y_thresh * errors
    cond1  = np.array(ytb > thresh, dtype='int')[1:]  # Raw Data S/N
    cond3  = np.array(d4y.copy()[1:] > 0.0, dtype='int')  # Positive 4nd derivative

    if (derv2_thresh > 0.):
        wsort   = np.argsort(np.abs(d2y))
        rms_d2y = ( np.std(d2y[wsort[0: int(0.5 * len(d2y))]]) / 0.377 )  # RMS based in +-1 sigma fluctuations
        thresh2 = -rms_d2y * derv2_thresh
    else:
        thresh2 = 0.
    # End-if
    
    cond4 = np.array(d2y.copy()[1:] < thresh2, dtype='int')  # Negative second derivative

    # Find optima of second derivative
    # --------------------------------
    zeros   = np.abs(np.diff(np.sign(d3y)))
    zeros   = zeros * cond1 * cond3 * cond4
    cen_id  = np.array(np.where(zeros)).ravel()  # Index cens
    cens    = fvel(cen_id + 0.5)            # Velocity cens
    N_gauss = len(cens)
    
    # print( 'Components found for alpha={1}: {0}'.format(N_gauss, SG_winlen) )

    # If nothing found, return null
    # ----------------------------------------------
    if N_gauss == 0:
        odict = {
            'cens': [],
            'FWHMs': [],
            'amps': [],
            'd2y': d2y,
            'errors': errors,
            'thresh2': thresh2,
            'thresh': thresh,
            'N_gauss': N_gauss
        }

        return odict


    # Find Relative widths, then measure
    # peak-to-inflection distance for sharpest peak
    wids  = np.sqrt(np.abs(ytb[cen_id] / d2y[cen_id]))
    FWHMs = wids * 2.355

    # Attempt deblending.
    # If Deblending results in all non-negative answers, keep.
    amps      = np.array(ytb[cen_id])
    FF_matrix = np.zeros([len(amps), len(amps)])
    for i in range(FF_matrix.shape[0]):
        for j in range(FF_matrix.shape[1]):
            FF_matrix[i, j] = np.exp(
                -(cens[i] - cens[j]) ** 2 / 2. / (FWHMs[j] / 2.355) ** 2
            )
    amps_new = np.linalg.lstsq(FF_matrix, amps, rcond=None)[0]
    if np.all(amps_new > 0):
        amps = amps_new

    odict = {
        'cens': cens,
        'FWHMs': FWHMs,
        'amps': amps,
        'd2y': d2y,
        'errors': errors,
        'thresh2': thresh2,
        'thresh': thresh,
        'N_gauss': N_gauss
    }

    return odict






## Do the fit: Gaussian Decomposition ##
 #
 # params 
 #
 # return 
 #
 # version 11/2019
 # author Nguyen Hiep ##
def fit(
    vel,
    ytb,
    errors,
    SG_winlen=None,
    SG_order=None,
    y_thresh=5.,
    band_frac=0.1,
    derv2_thresh=5.):


    dv = np.abs(vel[1] - vel[0])

    # --------------------------------------#
    # Find Gaussian components              #
    # --------------------------------------#
    gcpnts = find_peaks(
        vel,
        ytb,
        errors       = None,
        SG_winlen    = SG_winlen,
        SG_order     = SG_order,
        y_thresh     = y_thresh,
        band_frac    = band_frac,
        derv2_thresh = derv2_thresh
    )

    d2y       = gcpnts['d2y']
    par_guess = np.append(np.append(gcpnts['amps'], gcpnts['FWHMs']), gcpnts['cens'])
    ng_guess  = len(par_guess) // 3


    par_gfit  = par_guess
    ng_gfit   = len(par_gfit) // 3

    # Sort by amplitude
    hgt_temp   = par_gfit[0           : ng_gfit]
    wid_temp   = par_gfit[ng_gfit     : 2 * ng_gfit]
    cen_temp   = par_gfit[2 * ng_gfit : 3 * ng_gfit]
    idsort     = np.argsort(hgt_temp)[::-1]            # Sorting
    par_gfit   = np.concatenate(
                    [hgt_temp[idsort], wid_temp[idsort], cen_temp[idsort]]
                   )


    if (ng_gfit > 0):

        hgtlimd = [[True,True]]*ng_gfit
        cenlimd = [[False,False]]*ng_gfit
        widlimd = [[True,True]]*ng_gfit
        
        hgtlims = [[0., 999.]]*ng_gfit
        cenlims = [[False,False]]*ng_gfit
        widlims = [[0., 999.]]*ng_gfit
        
        hgtstep = [0.5]*ng_gfit
        censtep = [0.5]*ng_gfit
        widstep = [0.5]*ng_gfit
        
        plimd   = hgtlimd + cenlimd + widlimd
        plims   = hgtlims + cenlims + widlims
        pstep   = hgtstep + censtep + widstep

        parinfo = []
        for j in range( len(par_gfit) ):
            parbase = {'value': par_gfit[j], 'fixed': 0, 'parname': '', 'step':pstep[j], 'limited': plimd[j], 'limits': plims[j]}
            # parbase = {'value': guessp[j], 'fixed': None, 'parname': '', 'step':0, 'limited': [0, 0], 'limits': [0., 0.]}
            parinfo.append(cp.deepcopy(parbase))
        ## Endfor

        ##  MPFIT ##
        fa  = {'x':vel, 'y':ytb, 'err':errors}
        mp  = mpfit(func_mpfit, par_gfit, parinfo=parinfo, functkw=fa, maxiter=500, quiet=True)

        ## Params and their errors
        params_fit  = mp.params
        params_errs = mp.perror

        ncomps_fit  = len(params_fit) // 3


        yfit  = func(vel, *params_fit).ravel()
        rchi2 = np.sum((ytb - yfit) ** 2 / errors ** 2) / len(ytb)

        # Check if any amplitudes are identically zero, if so, remove them.
        if np.any(params_fit[0:ng_gfit] == 0):
            amps_fit   = params_fit[0           : ng_gfit]
            fwhms_fit  = params_fit[ng_gfit     : 2 * ng_gfit]
            cens_fit   = params_fit[2 * ng_gfit : 3 * ng_gfit]

            id_keep    = amps_fit > 0.0
            params_fit = np.concatenate(
                [amps_fit[id_keep], fwhms_fit[id_keep], cens_fit[id_keep]]
            )
            ncomps_fit = len(params_fit) // 3
    else:
        yfit       = np.zeros(len(vel))
        rchi2      = -999.
        ncomps_fit = ng_gfit
    # End - if ng_gfit > 0

    


    #                       P L O T T I N G
    if (False):
        ymax = np.max(ytb)
    
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12,6))

        # axis
        ax = axs[0]

        # Initial Guesses (Panel 1)
        d2y_scale = 1.0 / np.max(np.abs(d2y)) * ymax * 0.25
        ax.plot(vel, ytb, '-k', label='data')
        ax.plot(vel, d2y * d2y_scale, '--r', label='2nd derv')
        # ax1.plot(vel, vel / vel * gcpnts['thresh'], '-k')
        # ax1.plot(vel, vel / vel * gcpnts['thresh2'] * d2y_scale, '--r')
    
        for i in range(ng_guess):
            yc = gaussian(
                par_guess[i], par_guess[i + ng_guess], par_guess[i + 2 * ng_guess]
            )(vel)
            ax.plot(vel, yc, '-b', label='compnt')

        ax.set_title('Initial Guesses, ncomps = {0}'.format(ng_guess))
        ax.legend(loc=1)
    
        
        # Plot best-fit model (Panel 2)
        ax = axs[1]
        ax.plot(vel, ytb, label='data', color='black')
        for i in range(ncomps_fit):
            yc = gaussian(params_fit[i], params_fit[i + ncomps_fit], params_fit[i + 2 * ncomps_fit])(vel)
            ax.plot(vel, yc, '-', color='b')
        
        ax.plot(vel, yfit, '-r', label='Best fit')

        ax.set_title('Best fit, ncomps = {0}'.format(ncomps_fit), fontsize=16)
    
        ax.legend(loc=1)
        plt.show()
        plt.close()
    #  End -  P L O T T I N G

    



    # Output dictionary
    # -----------------------------------
    odict = {}
    odict['init_pars'] = par_gfit
    odict['N_gauss']   = ncomps_fit

    if (ncomps_fit > 0):
        odict['fit_pars'] = params_fit
        odict['fit_err']  = params_errs
        odict['rchi2']    = rchi2

    return (1, odict)