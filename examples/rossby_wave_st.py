import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import pyqg

# the model object
year = 1.
m = pyqg.BTModel(L=2.*pi,nx=128, tmax = 1*year,
        beta = 20., H = 1., rek = 0., rd = None, dt = 0.001,
                     taveint=year, ntd=4)
                     
#                      L is domain length
# 					   nx==ny is the number of grid points used in the x direction.
#                      Beta is the coriolis parameter gradient
#                      H is the depth of the fluid (Need to check this, but is it only used in the Enstrophy calculation???)
#                      rek is the friction coefficient
#                      rd is the Rossby deformation length
#                      taveint is the time at which the temporal averaging should start.

# Gaussian IC
fk = m.wv != 0 #s m.wv is the total wavenumber grid, i.e. ktot(i,j) in other language. By adding != 0 I'm guessing it returns an array of values where wv ne 0.
ckappa = np.zeros_like(m.wv2) #s makes an array of zeros the same size and shape as wv2.
ckappa[fk] = np.sqrt( m.wv2[fk]*(1. + (m.wv2[fk]/36.)**2) )**-1 #s makes some initial field that I don't think is currently used.

nhx,nhy = m.wv2.shape

R = pi/12.
Pi = -np.exp(-((m.x-3*pi/2)**2 + (m.y-pi)**2)/R**2) #some Gaussian blob

Pi = Pi - Pi.mean() #Make sure the gaussian blob has its mean subtracted.
Pi_hat = m.fft( Pi[np.newaxis,:,:] )
KEaux = m.spec_var(m.wv*Pi_hat ) #Work out the kinetic energy of the blob so that we can normalise it away. 

pih = ( Pi_hat/np.sqrt(KEaux) ) #Normalise by kinetic energy of blob.
qih = -m.wv2*pih #Get the gaussian blob from a streamfunction to a relative vorticity.
qi = m.ifft(qih) #Fourier transform the relative vorticity into physical space
m.set_q(qi) # Set the physical space field to be the blob.

# run the model
plt.rcParams['image.cmap'] = 'RdBu'

plt.ion()

for snapshot in m.run_with_snapshots(tsnapstart=0, tsnapint=10*m.dt):
    plt.clf()
    plt.imshow(m.q.squeeze())
    plt.clim([-20., 20.])
    plt.xticks([])
    plt.yticks([])

    plt.pause(0.01)
    plt.draw()
    plt.ioff()
    

