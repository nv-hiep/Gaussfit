import pickle
import os
import numpy             as np
import matplotlib.pyplot as plt

def gaussian(amp, fwhm, mean):
    return lambda x: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)

# Specify datfile of output data
datfile = 'data/EM_profile.pickle'


ngauss = 4
hgts  = [20., 11., 18., 15.]
fwhms = [20., 60., 40., 30.] # channels
cens  = [200, 320, 250, 360] # channels


# ngauss = 5
# hgts  = [20., 11., 18., 10., 15.]
# fwhms = [20., 60., 40., 20., 30.] # channels
# cens  = [220, 320, 260, 280, 360] # channels

# Data properties
noise = 0.25
nchan = 512

# Initialize
data     = {}
chan     = np.arange(nchan)
errors   = np.ones(nchan) * noise

spectrum = np.random.randn(nchan) * noise

# Create spectrum
for a, w, m in zip(hgts, fwhms, cens):
    spectrum += gaussian(a, w, m)(chan)


data['y']    = data.get('y', []) + [spectrum]
data['x']    = data.get('x', []) + [chan]
data['erry'] = data.get('erry', []) + [errors]

os.system('rm -rf ' + datfile)

pickle.dump(data, open(datfile, 'wb'))
print('Created: ', datfile)



if(True):
    plt.figure(figsize=(10,12))

    plt.plot(data['x'][0], data['y'][0] )

    for a, w, m in zip(hgts, fwhms, cens):
        spec = gaussian(a, w, m)(chan)
        plt.plot(data['x'][0], spec )

    plt.title(r'Data')
    plt.xlabel('Channels')
    # plt.xlabel('VLSR [km/s]')
    plt.ylabel('T [K]')
    plt.show()