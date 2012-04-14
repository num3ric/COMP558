from __future__ import division
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.animation as animation
import Image


image = numpy.array(Image.open("moo.png"))

P = (image - image.max() / 2) / 255
fig = plt.figure()
#plt.contour(P, levels=[0], colors='r')
im = plt.imshow(P, cmap=cm.gray)
dt = 0.01
F = 10

#plt.ion()

def updatefig(*args):
    global P
    # upwind scheme?
    [Px, Py] = numpy.gradient(P)
    P = P + dt * F * numpy.sqrt(Px**2 + Py**2)
    im.set_array(P)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()

# for i in range(1000):
#     [Px, Py] = numpy.gradient(P)
#     P += dt * F * numpy.sqrt(Px**2 + Py**2)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n = len(P)
# X, Y = numpy.mgrid[0:n, 0:n]
# ax.plot_surface(X, Y, P)

# plt.figure()
# plt.contour(P, levels=[0], colors='r')
# plt.imshow(P, interpolation='bilinear', cmap=cm.gray)
# plt.show()
