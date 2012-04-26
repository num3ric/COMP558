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

a = np.ones(constraint.shape)
a[constraint.shape[0]/2,constraint.shape[1]/2] = 0
P = ndimage.distance_transform_edt(a)
P -= 0.65*np.max(P)
# plt.figure()
# plt.imshow(P,cmap='gray')
# plt.contour(P, levels=[0])
# plt.show()
# P /= np.max(np.abs(P))

# P = (image - image.max() / 2) / 255
# fig = plt.figure()
#plt.contour(P, levels=[0], colors='r')
# im = plt.imshow(P, cmap=cm.gray)
dt = 0.5
F =constraint
fig = plt.figure()
plt.clf()
im = plt.imshow(P, cmap='jet')
plt.contour(P, levels=[0])
fig.colorbar(im)
plt.draw()

def lsm_grad_magnitude(f, up):
    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'
    out = np.zeros_like(f).astype(otype)
    for i in xrange(out.shape[0]):
        if up:
            if i > 0: #difference with preceding row
                out[i,:] = (out[i,:]) + np.maximum(f[i,:] - f[i - 1,:], 0)**2
            else: # difference with next row 'mirrored' outside the matrix
                out[i,:] = (out[i,:]) + np.maximum(f[i,:] - f[i + 1,:], 0)**2
            if i < out.shape[0] - 1: #difference from next row
                out[i,:] = out[i,:] + np.minimum(f[i + 1,:] - f[i,:], 0)**2
            else: # difference current row with next row 'mirrored' outside the matrix
                out[i,:] = out[i,:] + np.minimum(f[i - 1,:] - f[i,:], 0)**2
        else:
            if i > 0:
                out[i,:] = (out[i,:]) + np.minimum(f[i,:] - f[i - 1,:], 0)**2
            else:
                out[i,:] = (out[i,:]) + np.minimum(f[i,:] - f[i + 1,:], 0)**2
            if i < out.shape[0] - 1:
                out[i,:] = out[i,:] + np.maximum(f[i + 1,:] - f[i,:], 0)**2
            else:
                out[i,:] = out[i,:] + np.maximum(f[i - 1,:] - f[i,:], 0)**2
    for j in xrange(out.shape[1]):
        if up:
            if j > 0:
                out[:,j] = out[:,j] + np.maximum(f[:,j] - f[:,j - 1], 0)**2
            else:
                out[:,j] = out[:,j] + np.maximum(f[:,j] - f[:,j + 1], 0)**2
            if j < out.shape[1] - 1:
                out[:,j] = out[:,j] + np.minimum(f[:,j + 1] - f[:,j], 0)**2
            else:
                out[:,j] = out[:,j] + np.minimum(f[:,j - 1] - f[:,j], 0)**2
        else:
            if j > 0:
                out[:,j] = out[:,j] + np.minimum(f[:,j] - f[:,j - 1], 0)**2
            else:
                out[:,j] = out[:,j] + np.minimum(f[:,j] - f[:,j + 1], 0)**2
            if j < out.shape[1] - 1:
                out[:,j] = out[:,j] + np.maximum(f[:,j + 1] - f[:,j], 0)**2
            else:
                out[:,j] = out[:,j] + np.maximum(f[:,j - 1] - f[:,j], 0)**2
    return np.sqrt(out)

def update_phi(P, dt):
    # [Px, Py] = np.gradient(P)
    up = lsm_grad_magnitude(P, True)
    down = lsm_grad_magnitude(P, False)
    #to obtain correct results, down and up are switched
    return P + dt * down 

def updatefig(*args):
    global P
    P = update_phi(P, dt)
    im.set_array(P)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
#fig.savefig('test_plot')
