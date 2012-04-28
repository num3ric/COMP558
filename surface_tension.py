from __future__ import division
import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage
import skfmm
import sys
import itertools
import pickle

eps = sys.float_info.epsilon

wx, wy = 200, 200
grid_shape = (wx,wy)
cx, cy = wx/2, wy/2


def _space_grid(Sm, spacing):
    '''
    Zero everywhere, except for the points at a certain grid interval.
    '''
    Ix, Iy = np.indices(Sm.shape)
    return Sm * (np.fmod(Ix, spacing) == 0) * (np.fmod(Iy, spacing) == 0)

def _point_shape(dist_fn, radius, spacing, dist_metric=None):
    Sm = np.ones(grid_shape)
    Sm[cx,cy] = 0
    Sm = dist_fn(Sm, metric=dist_metric) if dist_metric else dist_fn(Sm)
    #retain only the points at the defined radius
    Sm = (Sm == radius) 
    return _space_grid(Sm, spacing)

def square(radius, spacing=1):
    '''
    Get a matrix defining a centered square.
    Zero everywhere except for points which have value 1.
    Note:   Since this uses the manhattan distance, the square
            rotated 45 degrees.
    '''
    return _point_shape(ndimage.distance_transform_cdt, radius, spacing, 'taxicab')


def circle(radius, spacing=1):
    '''
    Get a matrix defining a centered circle.
    Zero everywhere except for points which have value 1.
    '''
    return _point_shape(ndimage.distance_transform_edt, radius, spacing)

def divergence(U, V):
    '''
    Compute the divergence (scalar field) of the gradient vector field.
    '''
    [Uu, Uv] = np.gradient(U)
    [Vu, Vv] = np.gradient(V)
    return Uu + Vv

def gradient_magnitude(U, V):
    '''
    Compute the magnitude (scalar field) of the gradient vector field.
    '''
    return np.maximum(np.sqrt(U**2 + V**2), eps)

def lsm_grad_magnitude(f, up):
    '''
    The gradient magnitude using the upwind scheme.
    '''
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

# S = square(radius=25, spacing=5) # data set of points
S = np.array(Image.open("double1.png"))
S = ndimage.laplace(ndimage.gaussian_filter(S-0.5,1))
S = np.absolute(S < 30)
D = ndimage.distance_transform_edt(S) # distance to data set
[Du, Dv] = np.gradient(D)
# image = np.array(Image.open("cir.png"))
# Phi0 = (image - image.max() / 2) / 255

Phi0 = np.ones(grid_shape)
Phi0[cx, cy] = 0
Phi0 = ndimage.distance_transform_edt(Phi0)
P = Phi0 -  0.65*np.max(Phi0)

# plt.figure()
# plt.imshow(1-S, cmap='gray')
# fig = plt.figure()
# cax = plt.imshow(D)
# fig.colorbar(cax)
# plt.figure()
# plt.imshow(P,cmap='jet')
# plt.contour(P, levels=[0])

def compute_force(U, V, Gmag):
    '''
    Compute and return the force for the level set evolution.
    '''
    PF = (Du * U + Dv * V)/Gmag
    ST = D * divergence(U/Gmag, V/Gmag)
    return PF + ST

# [U, V] = np.gradient(Phi0)
# Gmag = gradient_magnitude(U, V)
# fig = plt.figure()
# im = plt.imshow((Du * U + Dv * V)/Gmag)
# fig.colorbar(im)


# unitF = np.ones(grid_shape)

def compute_stepsize(F):
    mf = np.max(np.absolute(F))
    safety_factor = 0.88
    return safety_factor / mf

def update_phi(P):
    '''
    Step update (euler method) the level set PDE.
    '''
    [U, V] = np.gradient(P)
    Gmag = gradient_magnitude(U, V)
    F = compute_force(U, V, Gmag)
    dt = compute_stepsize(F)
    return P + dt * F * Gmag
    


def update_phi_upwind(P):
    '''
    Step update (euler method) the level set PDE using the upwind scheme.
    '''
    [U, V] = np.gradient(P)
    Gmag = gradient_magnitude(U, V)
    F = compute_force(U, V, Gmag)
    dt = compute_stepsize(F)
    up = lsm_grad_magnitude(P, True)
    down = lsm_grad_magnitude(P, False)
    return P + dt *(np.maximum(F, 0)*down + np.minimum(F, 0)*up)

fig = plt.figure()

# plt.ion()
# for i in itertools.count():
#     if i % 100 == 0:
#         P = update_phi(P, dt)
#         plt.clf()
#         im = plt.imshow(P, cmap='jet')
#         plt.contour(P, levels=[0])
#         fig.colorbar(im)
#         plt.draw()


im = plt.imshow(P, cmap='jet')
fig.colorbar(im)
    
def save_figure(P, number=99):
    with open(str(number)+'p.pickle', 'w') as f:
        pickle.dump(skfmm.distance(P), f)

frame = 0
saved_frames = [0, 84, 220, 280]
def updatefig(*args):
    global P, frame
    for i in xrange(50):
        P = update_phi(P)
    P = skfmm.distance(P) #reinitialization
    im.set_array(P)
    # if frame % 50:
    #     save_figure(P)
    frame = frame + 1 
    return im,
ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

plt.show()
