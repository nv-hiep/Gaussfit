import sys, os
sys.path.insert(0, os.getenv('HOME')+'/projects/Gaussfit/modules') # add folder of Classes/modules

import pickle
import decomposer as dc

# Set parameters
datafile       = 'data/training_data.pickle'
y_thresh       = 5.
SG_winlen_init = 51
SG_order_init  = 7

d = dc.decomposer()

# Load training dataset
d.load_training_data(datafile)

# Set parameters
d.set('y_thresh', y_thresh)

# Train
d.train(SG_winlen_init = SG_winlen_init, SG_order_init = SG_order_init)