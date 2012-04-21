from __future__ import division
import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage
import sys
import itertools

eps = sys.float_info.epsilon

n, m = 200, 200
grid_shape = (n,m)
cx, cy = n/2, m/2


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


S = square(radius=50, spacing=10) # data set of points
D = ndimage.distance_transform_edt(1-S) # distance to data set
[Du, Dv] = np.gradient(D)
# image = np.array(Image.open("cir.png"))
# Phi0 = (image - image.max() / 2) / 255
Phi0 = np.ones(grid_shape)
Phi0[cx, cy] = 0
Phi0 = ndimage.distance_transform_edt(Phi0)
Phi0 -= 0.65*np.max(Phi0)

# plt.figure()
# plt.imshow(S)
# plt.figure()
# plt.imshow(D)
# plt.figure()
# plt.imshow(Phi0,cmap='gray')
# plt.contour(Phi0, levels=[0])

def compute_force(U, V, Gmag):
    '''
    Compute and return the force for the level set evolution.
    '''
    PF = (Du * U + Dv * V)/Gmag
    ST = D * divergence(U/Gmag, V/Gmag)
    return PF + ST

# unitF = np.ones(grid_shape)

def update_phi(P, dt):
    '''
    Step update (euler method) the level set PDE.
    '''
    [U, V] = np.gradient(P)
    Gmag = gradient_magnitude(U, V)
    F = compute_force(U, V, Gmag)
    return P + dt * F * Gmag

# def renormalize(P, dt): #not working...
#     [U, V] = np.gradient(P)
#     Gmag = gradient_magnitude(U, V)
#     S = P/np.sqrt(P*P + Gmag)
#     return P + dt*S * (1-Gmag)

dt = 0.01
P = Phi0
plt.ion()
fig = plt.figure()
for i in itertools.count():
    P = update_phi(P, dt)
    if i % 120 == 0:
        plt.clf()
        cax = plt.imshow(P, cmap='jet')
        plt.contour(P, levels=[0])
        fig.colorbar(cax)
        plt.draw()
    if i % 50 == 0:
        Pedt = ndimage.distance_transform_edt(np.round(P))
        P = np.sign(P) * Pedt
        # P = renormalize(P,1.0)

plt.show()
