# For training parameters SG_winlen and SG_order

import inspect
import fit
import signal

import multiprocessing  as mpc
import numpy            as np

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)



## Counts number of continuous trailing '1's
 # Used in the convergence criteria
 #
 # params 
 #
 # return 
 #
 # version -
 # author - ##
def count_ones_in_row(row):
    ret = np.zeros(len(row))
    for i in range(len(ret)):
        if (row[i] == 0):
            ret[i] = 0
        else:
            total = 1
            counter = 1
            while (row[i - counter] == 1):
                total += 1
                if (i - counter < 0):
                    break
                counter += 1

            ret[i] = total
            # end - while
        # end - if
    # end - for
    return ret




## Compare guess and true params
 #
 # params 
 #
 # return 
 #
 # version -
 # author - ##
def compare_pars(guess_params, true_params):
    ''' Figure of merit for comparing guesses to true components.
        guess_params = list of 3xN parameters for the N guessed Gaussians
                     = [amp1, amp2, amp3 ..., width1, width2, width3, ...
                            cen1, cen2, cen3]
        true_params  = list of 3xN parameters for the N true Gaussians '''

    # Extract parameters
    n_true  = len(true_params) // 3
    n_guess = len(guess_params) // 3
    
    guess_hgts    = guess_params[0:n_guess]
    guess_FWHMs   = guess_params[n_guess: 2 * n_guess]
    guess_cens    = guess_params[2 * n_guess: 3 * n_guess]
    
    true_hgts     = true_params[0:n_true]
    true_FWHMs    = true_params[n_true: 2 * n_true]
    true_cens     = true_params[2 * n_true: 3 * n_true]

    truth_matrix  = np.zeros([n_true, n_guess], dtype='int')
    # truth_matrix[i,j] = 1 if guess 'j' is a correct match to true component 'i'

    # Loop through answers and guesses
    for i in range(n_true):
        for j in range(n_guess):
            sigs_away = np.abs(
                (true_cens[i] - guess_cens[j]) / (true_FWHMs[i] / 2.355)
            )
            if (
                (sigs_away < 1.)
                and (guess_FWHMs[j] > 0.3 * true_FWHMs[i])  # for position
                and (guess_FWHMs[j] < 2.5 * true_FWHMs[i])  # Width match
                and (guess_hgts[j] > 0.)
                and (guess_hgts[j] < 5. * true_hgts[i])    # Amplitude match
            ):
                # Check make sure this guess/answer pair in unique
                if ( not 1 in np.append(truth_matrix[i,:], truth_matrix[:, j]) ):
                    truth_matrix[i, j] = 1
                # end - if
            # end - if
        # end - for
    # end - for

    # Compute this training example's recall and precision
    n_correct = float(np.sum(np.sum(truth_matrix)))

    return n_correct, n_guess, n_true





## Training function
 #
 # params 
 #
 # return 
 #
 # version -
 # author - ##
def training_fcn(kwargs):

    j = kwargs['j']
    true_params = np.append(
        kwargs['hgts'][j], np.append(kwargs['FWHMs'][j], kwargs['cens'][j])
    )

    # Produce initial guesses
    status, result = fit.fit(
        kwargs['vel'][j],
        kwargs['ytb'][j],
        kwargs['erry'][j],
        SG_winlen    = kwargs['SG_winlen'],
        SG_order     = kwargs['SG_order'],
        y_thresh     = kwargs['y_thresh'],
        derv2_thresh = kwargs['derv2_thresh']
    )

    # If nothing was found, skip to next iteration
    if (status == 0):
        print('Nothing found in this spectrum,  continuing...')
        return 0, 0, true_params // 3

    guess_params = result['init_pars']

    return compare_pars(guess_params, true_params)






## Cost-like function
 #
 # params 
 #
 # return 
 #
 # version -
 # author - ##
def cost_fcn(
    SG_winlen,
    SG_order,
    training_data,
    y_thresh     = 5.,
    derv2_thresh = 0.,
    ytb          = None,
    erry         = None,
    cens         = None,
    vel          = None,
    FWHMs        = None,
    hgts         = None
):

    # Obtain dictionary of current-scope keywords/arguments
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    del values['frame']  # This key not part of function arguments

    # Construct iterator of dictionaries of keywords for multi-processing
    mp_params = iter(
        [
            dict(list(values.items()) + list({'j': j}.items()))
            for j in range(len(training_data['y']))
        ]
    )

    # Multiprocessing code
    ncpus = mpc.cpu_count()
    pl    = mpc.Pool(ncpus, init_worker)


    try:
        mp_results = pl.map(training_fcn, mp_params)

    except KeyboardInterrupt:
        print('KeyboardInterrupt... quitting.')
        pl.terminate()
        quit()

    pl.close()
    del pl

    Nc, Ng, Nt = np.array(mp_results).sum(0)
    accuracy = 2. * Nc / (Ng + Nt)  # Cumulative accuracy

    return -np.log(accuracy)




## A class to store variables ##
 #
 # params 
 #
 # return 
 #
 # version 11/2019
 # author Nguyen Hiep ##
class store(object):
    def __init__(self, iters):
        self.SG_winlen_hist      = np.zeros(iters + 1) * np.nan
        self.SG_order_hist       = np.zeros(iters + 1) * np.nan
        self.accuracy_hist       = np.zeros(iters)     * np.nan
        self.D_SG_winlen_hist    = np.zeros(iters)     * np.nan
        self.D_SG_order_hist     = np.zeros(iters)     * np.nan
        self.SG_winlenmeans1     = np.zeros(iters)     * np.nan
        self.SG_winlenmeans2     = np.zeros(iters)     * np.nan
        self.SG_ordermeans1      = np.zeros(iters)     * np.nan
        self.SG_ordermeans2      = np.zeros(iters)     * np.nan
        self.fracdiff_SG_winlen  = np.zeros(iters)     * np.nan
        self.fracdiff_SG_order   = np.zeros(iters)     * np.nan
        self.it_converge         = np.nan







## For training using gradient descent with momentum (gamma parameter)
 #
 # params 
 #
 # return 
 #
 # version -
 # author - ##
def train(
    cost_fcn           = cost_fcn,
    training_data      = None,
    SG_winlen_init     = None,
    SG_order_init      = None,
    iters              = 500,
    epsilon            = None,
    learning_rate      = None,
    gamma              = None,
    window_size        = 5,     # 10
    iters4convrg       = 10,
    derv2_thresh       = 0.,
    y_thresh           = 5.
):
    '''
    SG_winlen_init =
    SG_order_init =
    iters =
    epsilon = 'epsilon; finite winlen step for computing derivatives in gradient'
    learning_rate
    gamma = 'Momentum value'
    window_size = trailing window size to determine convergence,
    iters4convrg = number of continuous iters
        within threshold tolerence required to acheive convergence
    '''

    # Some parameters
    if (not learning_rate):
        learning_rate = 0.5 # 0.9
    
    if (not epsilon):
        epsilon = 0.25

    if (not gamma):
        gamma = 0.2

    # threshold for SG_winlen and SG_order
    thresh = 7


    # Training data
    ytb    = training_data['y']
    vel    = training_data['x']
    erry   = training_data['erry']
    cens   = training_data['cens']
    FWHMs  = training_data['wids']
    hgts   = training_data['hgts']

    # to store params
    sg = store(iters)
    sg.SG_winlen_hist[0] = SG_winlen_init
    sg.SG_order_hist[0]  = SG_order_init

    for i in range(iters):
        # For winlen
        SG_winlen_r, SG_winlen_c, SG_winlen_l = (
            sg.SG_winlen_hist[i] + epsilon,
            sg.SG_winlen_hist[i],
            sg.SG_winlen_hist[i] - epsilon,
        )

        # For order
        SG_order_r, SG_order_c, SG_order_l = (
            sg.SG_order_hist[i] + epsilon,
            sg.SG_order_hist[i],
            sg.SG_order_hist[i] - epsilon,
        )

        # cost function
        cost_winlen_r = cost_fcn(
            SG_winlen_r,
            SG_order_c,
            training_data,
            ytb          = ytb,
            erry         = erry,
            cens         = cens,
            vel          = vel,
            FWHMs        = FWHMs,
            hgts         = hgts,
            y_thresh     = y_thresh,
            derv2_thresh = derv2_thresh
        )



        if (epsilon == 0.):
            print('Mean Accuracy: ', np.exp(-cost_winlen_r))
            quit()
        

        cost_winlen_l = cost_fcn(
            SG_winlen_l,
            SG_order_c,
            training_data,
            ytb          = ytb,
            erry         = erry,
            cens         = cens,
            vel          = vel,
            FWHMs        = FWHMs,
            hgts         = hgts,
            y_thresh     = y_thresh,
            derv2_thresh = derv2_thresh
        )

        sg.D_SG_winlen_hist[i] = (cost_winlen_r - cost_winlen_l) / 2. / epsilon
        sg.accuracy_hist[i]    = (cost_winlen_r + cost_winlen_l) / 2.

        
        cost_order_r = cost_fcn(
            SG_winlen_c,
            SG_order_r,
            training_data,
            ytb          = ytb,
            erry         = erry,
            cens         = cens,
            vel          = vel,
            FWHMs        = FWHMs,
            hgts         = hgts,
            y_thresh     = y_thresh,
            derv2_thresh = derv2_thresh
        )

        cost_order_l = cost_fcn(
            SG_winlen_c,
            SG_order_l,
            training_data,
            ytb          = ytb,
            erry         = erry,
            cens         = cens,
            vel          = vel,
            FWHMs        = FWHMs,
            hgts         = hgts,
            y_thresh     = y_thresh,
            derv2_thresh = derv2_thresh
        )

        sg.D_SG_order_hist[i] = (cost_order_r - cost_order_l) / 2. / epsilon
        sg.accuracy_hist[i]   = (cost_winlen_r + cost_winlen_l + cost_order_r + cost_order_l) / 4.

        if (i == 0):
            momentum1 = 0.
            momentum2 = 0.
        else:
            momentum1 = gamma * (sg.SG_winlen_hist[i] - sg.SG_winlen_hist[i - 1])
            momentum2 = gamma * (sg.SG_order_hist[i] - sg.SG_order_hist[i - 1])
        # End - if

        sg.SG_winlen_hist[i + 1] = ( sg.SG_winlen_hist[i] -
                                     learning_rate * sg.D_SG_winlen_hist[i] + momentum1 )
        sg.SG_winlen_hist[i + 1] = int( sg.SG_winlen_hist[i + 1] )

        
        sg.SG_order_hist[i + 1]  = ( sg.SG_order_hist[i] -
                                     learning_rate * sg.D_SG_order_hist[i] + momentum2 )
        sg.SG_order_hist[i + 1]  = int( sg.SG_order_hist[i + 1] )


        print('')
        print(sg.SG_winlen_hist[i], learning_rate, sg.D_SG_winlen_hist[i], momentum1)
        print(
            'iter {0}: F1={1:4.1f}%, pars=[{2}, {3}], p=[{4:4.2f}, {5:4.2f}]'.format(
                i,
                100 * np.exp(-sg.accuracy_hist[i]),
                np.round(sg.SG_winlen_hist[i], 2),
                np.round(sg.SG_order_hist[i], 2),
                np.round(momentum1, 2),
                np.round(momentum2, 2),
            ),
            end=' ',
        )

        #    if False: (use this to avoid convergence testing)
        if (i <= 2 * window_size):
            print(' (Convergence testing begins in {} iterations)'.format(
                    int(2 * window_size - i) ))
        else:
            sg.SG_winlenmeans1[i] = np.mean(sg.SG_winlen_hist[i - window_size: i])
            sg.SG_winlenmeans2[i] = np.mean(sg.SG_winlen_hist[i - 2 * window_size: i - window_size])
            
            sg.SG_ordermeans1[i] = np.mean(sg.SG_order_hist[i - window_size: i])
            sg.SG_ordermeans2[i] = np.mean(sg.SG_order_hist[i - 2 * window_size: i - window_size])

            sg.fracdiff_SG_winlen[i] = np.abs(sg.SG_winlenmeans1[i] - sg.SG_winlenmeans2[i])
            sg.fracdiff_SG_order[i]  = np.abs(sg.SG_ordermeans1[i] - sg.SG_ordermeans2[i])

            converge_logic = (sg.fracdiff_SG_winlen < thresh) & (sg.fracdiff_SG_order < thresh)

            c = count_ones_in_row(converge_logic)
            print(
                '  ({0:4.2F},{1:4.2F} < {2:4.2F} for {3} iters [{4} required])'.format(
                    sg.fracdiff_SG_winlen[i],
                    sg.fracdiff_SG_order[i],
                    thresh,
                    int(c[i]),
                    iters4convrg,
                )
            )

            if np.any(c > iters4convrg):
                i_converge     = np.min(np.argwhere(c > iters4convrg))
                sg.it_converge = i_converge
                print('Convergence achieved at iteration: ', i_converge)
                break
        # End - if  <= 2 * window_size



    # Best params
    sg.SG_winlenmeans1[i] = int( np.floor(sg.SG_winlenmeans1[i]) )
    sg.SG_ordermeans1[i]  = int( np.floor(sg.SG_ordermeans1[i]) )

    if ( sg.SG_winlenmeans1[i] % 2 != 1 ):
        sg.SG_winlenmeans1[i] = sg.SG_winlenmeans1[i] - 1

    return sg.SG_winlenmeans1[i], sg.SG_ordermeans1[i], sg






if __name__ == '__main__':
    pass