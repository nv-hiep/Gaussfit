import sys, os
sys.path.insert(0, os.getenv('HOME')+'/projects/Gaussfit/modules') # add folder of Classes and modules

import pickle

import decomposer        as dc
import numpy             as np



# optimal parameters from training
SG_winlen = 35
SG_order  = 3
y_thresh  = 5.


data_file   = 'data/EM_profile_GNOMES_13T.pickle'
data_file   = 'data/EM_profile_GNOMES_13R.pickle'
data_file   = 'data/EM_profile_GNOMES_29T.pickle'
data_file   = 'data/EM_profile.pickle'

fit_results = 'data/EM_fit_results.pickle'


if ( os.path.isfile(fit_results) ):
	os.system('rm -rf ' + fit_results)

# Create object
d = dc.decomposer()

# Parameters
d.set('SG_winlen', SG_winlen)
d.set('SG_order', SG_order)
d.set('y_thresh', y_thresh)

# Run
res = d.parallel_fit(data_file)

# Save results
pickle.dump(res, open(fit_results, 'wb'))

dat  = pickle.load(open(fit_results, 'rb'), encoding='latin1')
spec = pickle.load(open(data_file, 'rb'), encoding='latin1')

print(dat)

print( dat['idx'] )

# Plot results
d._plot(spec, dat, 0, guesses=True)