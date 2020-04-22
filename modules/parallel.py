import pickle
import multiprocessing

import numpy      as np
import decomposer as dc

from concurrent.futures import ProcessPoolExecutor



## Load data, and prepare to fit ##
 #
 # params 
 #
 # return 
 #
 # version -
 # author Gausspy ##  
def init(*args):
    global xobj, data_file, idx_list, data

    if (args):
        [xobj, data] = args[0]
    else:
        [xobj, data_file] = pickle.load(
            open('data/xtemp.pickle', 'rb'), encoding='latin1'
        )
        data = pickle.load(open(data_file, 'rb'), encoding='latin1')
    # End - if
    
    idx_list = np.arange(len(data['y']))






## Fit the data and get results ##
 #
 # params 
 #
 # return 
 #
 # version 11/2019
 # author Nguyen Hiep ##  
def to_decompose(i):
    print('   ---->  ', i)
    result = dc.decomposer.to_fit(
        xobj,
        data['x'][i],
        data['y'][i],
        data['erry'][i],
    )
    return result






## Run the fit with parallel processes ##
 #
 # params 
 #
 # return 
 #
 # version -
 # author Gausspy ##  
def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=1):
    '''A parallel version of the map function with a progress bar.
    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the array elements
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements
            as dictionaries of keyword arguments to function
        front_num (int, default=3): The number of iterations to run serially
            before kicking off the parallel job. Useful for catching bugs
    Returns:
        [function(array[0]), function(array[1]), ...]
    '''
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [
            function(**a) if use_kwargs else function(a) for a in array[:front_num]
        ]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [
            function(**a) if use_kwargs else function(a)
            for a in array[front_num:]
        ]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
    

    # Get the results from the futures.
    out = []
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out






## Function to run parallel computation ##
 #
 # params 
 #
 # return 
 #
 # version -
 # author Gausspy ##  
def func(use_ncpus=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    if (use_ncpus is None):
        use_ncpus = int(0.75 * ncpus)
    

    print('using {} out of {} cpus'.format(use_ncpus, ncpus))
    
    try:
        results_list = parallel_process(idx_list, to_decompose, n_jobs=use_ncpus)
    except KeyboardInterrupt:
        print('KeyboardInterrupt... quitting.')
        quit()
    return results_list