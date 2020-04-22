import pickle
import os
import random
import numpy             as np
import matplotlib.pyplot as plt


def gaussian(amp, fwhm, mean):
    return lambda x: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)

# Number of channels (nchan)
nchan = 512

# number of spectra (nspec)
nspec = 200

# Uncertainty per channel (noise)
noise_lims = [0.7, 0.9]

# Range of Gaussian paramters
hgt_lims = [5., 40.]
wid_lims = [20, 80] # FWHM in channels
cen_lims = [0.25*nchan, 0.75*nchan] # channels [0.25*nchan, 0.75*nchan]

# Indicate whether the data created here will be used as a training set
# (a.k.a. decide to store the "true" answers or not at the end)
train_yn = True

# Specify the pickle file to store the results in
file = 'data/training_data.pickle'




# Create training dataset with Gaussian profiles -cont-

# Initialize
data = {}
chan = np.arange(nchan)

# Begin populating data
for i in range(nspec):
    noise  = np.random.uniform(noise_lims[0], noise_lims[1])
    erry   = np.ones(nchan) * noise
    ng     = random.randint(3, 5)
    
    spec_i = np.random.randn(nchan) * noise

    
    hgts = []
    wids = []
    cens = []
    for j in range(ng):
        # Select random values for components within specified ranges
        a = np.random.uniform(hgt_lims[0], hgt_lims[1])
        w = np.random.uniform(wid_lims[0], wid_lims[1])
        c = np.random.uniform(cen_lims[0], cen_lims[1])

        # Add Gaussian profile with the above random parameters to the spectrum
        spec_i += gaussian(a, w, c)(chan)

        # Append the parameters to initialized lists for storing
        hgts.append(a)
        wids.append(w)
        cens.append(c)

    # Enter results into AGD dataset
    data['y']    = data.get('y', [])    + [spec_i]
    data['x']    = data.get('x', [])    + [chan]
    data['erry'] = data.get('erry', []) + [erry]

    # If training data, keep answers
    if (train_yn):
        data['hgts'] = data.get('hgts', []) + [hgts]
        data['wids'] = data.get('wids', []) + [wids]
        data['cens'] = data.get('cens', []) + [cens]

if(True):
	# Dump synthetic data into specified filename
	pickle.dump(data, open(file, 'wb'))


if(True):
	dat = pickle.load(open(file, "rb"), encoding="latin1")

	for i in range(nspec):
	    plt.figure(figsize=(10,12))

	    plt.plot(dat['x'][i], dat['y'][i] )

	    for (a, w, c) in zip(dat['hgts'][i], dat['wids'][i], dat['cens'][i]):
	        spec_i = gaussian(a, w, c)(chan)
	        plt.plot(dat['x'][i], spec_i )

	    plt.title(r'Training data')
	    plt.xlabel('Channels')
	    # plt.xlabel('VLSR [km/s]')
	    plt.ylabel('T [K]')
	    plt.show()