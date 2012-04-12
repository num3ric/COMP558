# import matplotlib.pyplot as plt 
# import numpy as np 
# x, y= np.arange(0,2*np.pi,.2), np.arange(0,2*np.pi,.2) 
# X,Y = np.meshgrid(x,y) 
# U,V = np.cos(X), np.sin(Y) 
# plt.pcolor(X,Y,U**2+V**2) 
# plt.quiver(X,Y,U,V) 
# plt.show() 



# Example we might want to use as a reference to ANIMATE the level set evolution

# from pylab import *
# import time
# ion()
# tstart = time.time()               # for profiling
# x = arange(0,2*pi,0.01)            # x-array
# line, = plot(x,sin(x))
# for i in arange(1,200):
#     line.set_ydata(sin(x+i/10.0))  # update the data
#     draw()                         # redraw the canvas
# print 'FPS:' , 200/(time.time()-tstart)


from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import math

area_width = 15

bbox=(-2.5,2.5)

def plot_implicit_2d(fn):
	global bbox
	xmin, xmax, ymin, ymax = bbox*2
	fig = plt.figure()
	ax = fig.add_subplot(111)
	A = np.linspace(xmin,xmax, area_width)
	X,Y = np.meshgrid(A, A)

	for z in A: # plot contours in the XY plane
		Z = fn(X,Y)
		cset = ax.contour(X, Y, Z)

	plt.show()

def plot_implicit_3d(fn):
	''' create a plot of an implicit function
	fn  ...implicit function (plot where fn==0)
	bbox ..the x,y,and z limits of plotted interval'''
	global bbox
	xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	A = np.linspace(xmin, xmax, 100) # resolution of the contour
	B = np.linspace(xmin, xmax, 15) # number of slices
	A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

	for z in B: # plot contours in the XY plane
		X,Y = A1,A2
		Z = fn(X,Y,z)
		cset = ax.contour(X, Y, Z+z, [z], zdir='z')
		# [z] defines the only level to plot for this contour for this value of z

	for y in B: # plot contours in the XZ plane
		X,Z = A1,A2
		Y = fn(X,y,Z)
		cset = ax.contour(X, Y+y, Z, [y], zdir='y')

	for x in B: # plot contours in the YZ plane
		Y,Z = A1,A2
		X = fn(x,Y,Z)
		cset = ax.contour(X+x, Y, Z, [x], zdir='x')

	# must set plot limits because the contour will likely extend
	# way beyond the displayed level.  Otherwise matplotlib extends the plot limits
	# to encompass all values in the contour.
	ax.set_zlim3d(zmin,zmax)
	ax.set_xlim3d(xmin,xmax)
	ax.set_ylim3d(ymin,ymax)
	plt.show()

def goursat_tangle(X, Y, Z):
	a,b,c = 0.0,-5.0,11.8
	return X**4+Y**4+Z**4+a*(X**2+Y**2+Z**2)**2+b*(X**2+Y**2+Z**2)+c

def circle(X, Y, centerX=0, centerY=0, radius = 1):
	return (centerX-X)**2 + (centerY-Y)**2 - radius**2

def square(X, Y, centerX=0, centerY=0, radius = 1):
	return np.maximum(np.absolute(centerX-X), np.absolute(centerY - Y)) - radius

def step_function(X, Y, percent_width, height=1):
	width = X.shape[0]
	middle = int(width/2)
	rad = int(math.floor(percent_width*0.5*width))
	Z = -height*np.ones(X.shape)
	Z[middle-rad:middle+rad, middle-rad:middle+rad] = height
	return Z

# Test plots
# plot_implicit_3d(goursat_tangle)
# plot_implicit_2d(circle)

def get_explicit_level_points(fn, threshold = 0.03, centerX = 0, centerY=0, radius = 1):
	global bbox
	xmin, xmax, ymin, ymax = bbox*2
	A = np.linspace(xmin,xmax, 100)
	X,Y = np.meshgrid(A, A)
	Z = fn(X,Y, centerX=centerX, centerY=centerY, radius=radius)
	#create a mask array and preserve only the elements within threshold
	#i.e. only the elements where fn ~= 0
	X = X[np.absolute(Z) < threshold] 
	Y = Y[np.absolute(Z) < threshold]
	return X, Y

def get_psi_level_points(X, Y, psi, threshold = 0.03):
	X = X[np.absolute(psi) < threshold] 
	Y = Y[np.absolute(psi) < threshold]
	return X, Y

def evolve_interface(psi):
	'''
	TODO: Lots of work here to complete this function!
	'''
	dX, dY = np.gradient(psi)
	print dX
	print dY
	M = np.sqrt(np.vdot(dX, dX)+np.vdot(dY, dY))
	print M
	return psi + 0.1*M

A = np.linspace(bbox[0],bbox[1], area_width)
X,Y = np.meshgrid(A, A)
# psiZ = step_function(psiX, psiY, 0.5)

psi0 = circle(X, Y, radius=2)
psi1 = np.array(psi0)
# X, Y = get_explicit_level_points(square, centerX=-1, centerY=0, radius= 0.5)
# plt.plot(X,Y, 'bo')
# X, Y = get_explicit_level_points(circle, centerX=1, centerY=0, radius= 0.5)
# plt.plot(X,Y, 'bo')
# X, Y = get_explicit_level_points(circle, centerX=0, centerY=0, radius= 2.0)
# plt.plot(X,Y, 'bo')

psi1 = evolve_interface(psi0)
# print "psi0"
# print psi0
# print "psi1"
# print psi1
# X, Y = get_psi_level_points(X, Y, psi1)
# plt.plot(X, Y,'ro')
print np.shape(X), np.shape(Y), np.shape(psi0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, psi0, rstride=1, cstride=1, cmap=cm.jet,
#         linewidth=0, antialiased=False)
# ax.plot_surface(X, Y, psi1, rstride=1, cstride=1, cmap=cm.jet,
#         linewidth=0, antialiased=False)

ax.contour(X, Y, psi0)
ax.contour(X, Y, psi1)

plt.show()

