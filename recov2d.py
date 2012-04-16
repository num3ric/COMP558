from __future__ import division
import Image
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import sys
import itertools

eps = sys.float_info.epsilon

im = numpy.array(Image.open("cir.png"))
im = im / im.max()
# plt.figure()
# plt.imshow(im, cmap='gray')

# Sample down the edges to short segments
pts = numpy.sign(im * (im < 0.2) * (im > 0.05))
# plt.figure()
# plt.imshow(pts, cmap='gray')

# Compute the EDT
edt_im, indices = ndimage.distance_transform_edt(1 - pts, return_indices=True)
# plt.figure()
# plt.imshow(edt_im)#, cmap='gray')

# Compute the SDET
n, m = edt_im.shape
X, Y = indices - numpy.mgrid[0:n, 0:m]
# fig = plt.figure()
# plt.subplot(121)
# cax_x = plt.imshow(X)
# plt.contour(im, levels=[0.1], colors='r')
# fig.colorbar(cax_x)
# plt.subplot(122)
# cax_y = plt.imshow(Y)
# plt.contour(im, levels=[0.1])
# fig.colorbar(cax_y)

# plt.figure()
# plt.quiver(X, Y)


A=(numpy.sqrt(X ** 2 + Y ** 2) - edt_im)
print A.max(), A.min()

# Create initial implicit curve
# width = 35
# phi = numpy.ones(im.shape)
# phi[width-1:-width+1, width-1:-width+1] = 0
# phi[width:-width, width:-width] = -1

phi = numpy.ones(im.shape)
n, m = phi.shape
phi[n/2, m/2] = 0
phi = numpy.round(ndimage.distance_transform_edt(phi) - 5*n/12)

plt.figure()
plt.imshow(phi, cmap='gray', interpolation='none')
plt.contour(phi, levels=[0])
plt.show()

print (phi==0).any()

# Make phi a signed distance function
phi_edt = ndimage.distance_transform_edt(phi)
phi0 = numpy.sign(phi) * phi_edt
# fig = plt.figure()
# plt.subplot(121)
# cax_p0 = plt.imshow(phi0)
# fig.colorbar(cax_p0)
# plt.contour(phi0, levels=[0], colors='r')
# ax = fig.add_subplot(122, projection='3d')
# n = len(phi0)
# X, Y = numpy.mgrid[0:n, 0:n]
# ax.plot_surface(X, Y, phi0,  cmap='jet')

print (phi0 > 0).any()

def abs_gradient(array):
    [u, v] = numpy.gradient(array)
    return numpy.sqrt(u**2 + v**2)

Dphi0 = abs_gradient(phi0)
# plt.figure()
# plt.imshow(Dphi0)

print Dphi0.mean()
print numpy.median(Dphi0)

def div(u, v):
    [uu, uv] = numpy.gradient(u)
    [vu, vv] = numpy.gradient(v)
    return uu + vv

im = ndimage.gaussian_filter(im, 3)
[im_u, im_v] = numpy.gradient(im)
# plt.figure()
# plt.subplot(121)
# plt.imshow(div(im_u, im_v))
# plt.subplot(122)
# plt.imshow(ndimage.laplace(im))

# Compute the mean curvature K
im_g = numpy.sqrt(im_u**2 + im_v**2)
d = numpy.maximum(im_g, eps)
K = d * div(im_u / d, im_v / d)
# plt.figure()
# plt.subplot(121)
# plt.imshow(im, cmap='gray')
# plt.subplot(122)
# plt.imshow(K)

dt = 0.5
p = phi0

i = 0

def update_phi(p, dt):
    [pu, pv] = numpy.gradient(p)
    abs_grad_p = numpy.maximum(numpy.sqrt(pu**2 + pv**2), eps)
    PF = (im_u * pu + im_v * pv) / abs_grad_p
    ST = d * div(pu / abs_grad_p, pv / abs_grad_p)
    return p + dt * (PF + ST) * abs_grad_p

plt.ion()
fig = plt.figure()
for i in itertools.count():
    p = update_phi(p, dt)
    if i % 50 == 0:
        plt.clf()
        cax = plt.imshow(p, cmap='jet')
        plt.contour(p, levels=[0])
        fig.colorbar(cax)
        plt.draw()

# def updatefig(*args):
#     global p, i
#     i += 1
#     p = update_phi(p, dt)
#     im.set_array(p)
#     return im,
#
# fig = plt.figure()
# im = plt.imshow(p, cmap='gray')
# ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()

# plt.show()
