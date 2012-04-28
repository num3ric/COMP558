from __future__ import division
import Image
import numpy as np
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from mayavi import mlab
from scipy import ndimage
import sys
import itertools

'''
Warning: unstable evolution.
Please refer to final_lsm.py for a better version (only in 2d.)
'''

eps = sys.float_info.epsilon

cube = 100
grid_shape = (cube,cube,cube)
cx = cy = cz = cube/2


def _space_grid(Sm, spacing):
    '''
    Zero everywhere, except for the points at a certain grid interval.
    '''
    Ix, Iy, Iz = np.indices(Sm.shape)
    return Sm * (np.fmod(Ix, spacing) == 0) * (np.fmod(Iy, spacing) == 0) * \
           (np.fmod(Iz, spacing) == 0)

def _point_shape(dist_fn, radius, spacing, dist_metric=None):
    Sm = np.ones(grid_shape)
    Sm[cx,cy,cz] = 0
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

def divergence(U, V, W):
    '''
    Compute the divergence (scalar field) of the gradient vector field.
    '''
    [Uu, Uv, Uw] = np.gradient(U)
    [Vu, Vv, Vw] = np.gradient(V)
    [Wu, Wv, Ww] = np.gradient(V)
    return Uu + Vv + Ww

def gradient_magnitude(U, V, W):
    '''
    Compute the magnitude (scalar field) of the gradient vector field.
    '''
    return np.maximum(np.sqrt(U**2 + V**2 + W**2), eps)


S = square(radius=40, spacing=2) # data set of points
D = ndimage.distance_transform_edt(1-S) # distance to data set
[Du, Dv, Dw] = np.gradient(D)

mlab.contour3d(D,contours=[0.33],opacity=.8 )
# mlab.quiver3d(Du, Dv, Dw, mask_points=50, opacity=.2)

Phi0 = np.ones(grid_shape)
Phi0[cx, cy, cz] = 0
Phi0 = ndimage.distance_transform_edt(Phi0)
Phi0 -= 0.5*np.max(Phi0)

def compute_force(U, V, W, Gmag):
    '''
    Compute and return the force for the level set evolution.
    '''
    PF = (Du * U + Dv * V + Dw * W)/Gmag
    ST = D * divergence(U/Gmag, V/Gmag, W/Gmag)
    return PF + ST

# unitF = np.ones(grid_shape)

def update_phi(P, dt):
    '''
    Step update (euler method) the level set PDE.
    '''
    [U, V, W] = np.gradient(P)
    Gmag = gradient_magnitude(U, V, W)
    F = compute_force(U, V, W, Gmag)
    return P + dt * F * Gmag

dt = 0.01
P = Phi0

c3d = mlab.contour3d(P,contours=[0],opacity=.2 )

for i in itertools.count():
    P = update_phi(P, dt)
    if i % 20 == 0:
        c3d.mlab_source.scalars = P
    if i % 50 == 0:
        Pedt = ndimage.distance_transform_edt(np.round(P))
        P = np.sign(P) * Pedt
        # P = renormalize(P,1.0)


mlab.outline()  
mlab.axes()
mlab.show()
