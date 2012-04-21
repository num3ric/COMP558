from __future__ import division
import numpy as np
import matplotlib as mpl
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy import ndimage
import Image


image = np.array(Image.open("cir.png"))
constraint = np.array(Image.open("curve.png"))
#constraint = ndimage.filters.laplace(constraint)#ndimage.filters.gaussian_filter(constraint, sigma=0.2))
constraint /= np.max(np.absolute(constraint))
#fig = plt.figure()
#im = plt.imshow(constraint)

# a = np.ones(constraint.shape)
# a[constraint.shape[0]/2,constraint.shape[1]/2] = 0
# P = ndimage.distance_transform_edt(a)
# P -= 0.65*np.max(P)
# plt.figure()
# plt.imshow(P,cmap='gray')
# plt.contour(P, levels=[0])
# plt.show()
# P /= np.max(np.abs(P))

P = (image - image.max() / 2) / 255
# fig = plt.figure()
#plt.contour(P, levels=[0], colors='r')
# im = plt.imshow(P, cmap=cm.gray)
dt = 1
F =constraint
fig = plt.figure()
plt.clf()
im = plt.imshow(P, cmap='jet')
plt.contour(P, levels=[0])
# fig.colorbar(im)
plt.draw()

def update_phi(P, dt):
	[Px, Py] = np.gradient(P)
	P += dt * F * np.sqrt(Px**2 + Py**2)
	return P

def updatefig(*args):
    global P
    P = update_phi(P, dt)
    im.set_array(P)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
#fig.savefig('test_plot')
