import numpy as np
import sys

eps = sys.float_info.epsilon

def gradient(f, *varargs):
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == N:
        dx = list(varargs)
    else:
        raise SyntaxError(
                "invalid number of arguments")

    # use central differences on interior and first differences on endpoints

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range(N):
        # select out appropriate parts for this dimension
        out = np.zeros_like(f).astype(otype)
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(2, None)
        slice3[axis] = slice(None, -2)
        # 1D equivalent -- out[1:-1] = (f[2:] - f[:-2])/2.0
        out[slice1] = (f[slice2] - f[slice3])/2.0
        slice1[axis] = 0
        slice2[axis] = 1
        slice3[axis] = 0
        # 1D equivalent -- out[0] = (f[1] - f[0])
        out[slice1] = (f[slice2] - f[slice3])
        slice1[axis] = -1
        slice2[axis] = -1
        slice3[axis] = -2
        # 1D equivalent -- out[-1] = (f[-1] - f[-2])
        out[slice1] = (f[slice2] - f[slice3])

        # divide by step size
        outvals.append(out / dx[axis])

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)

    if N == 1:
        return outvals[0]
    else:
        return outvals


def upwind_gradient2d(f, a):
    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'
    outu = np.zeros_like(f).astype(otype)
    outv = np.copy(outu)
    for x in xrange(f.shape[0]):
        for y in xrange(f.shape[1]):
            if a[x,y] < 0:
                if x == outu.shape[0]-1:
                    outu[x,y] = f[x,y] - f[x-1, y]
                else:
                    outu[x,y] = f[x+1,y] - f[x,y]
            elif a[x,y] > 0:
                if x == 0:
                    outu[x,y] = f[x+1,y] - f[x,y]
                else:
                    outu[x,y] = f[x,y] - f[x-1, y]
            else:
                if x == outu.shape[0]-1:
                    outu[x,y] = f[x,y] - f[x-1, y]
                elif x == 0:
                    outu[x,y] = f[x+1,y] - f[x,y]
                else:
                    outu[x,y] = f[x+1,y] - f[x-1, y]

            if a[x,y] < 0:
                if y == outv.shape[0]-1:
                    outv[x,y] = f[x,y] - f[x, y-1]
                else:
                    outv[x,y] = f[x,y+1] - f[x,y]
            elif a[x,y] > 0:
                if y == 0:
                    outv[x,y] = f[x,y+1] - f[x,y]
                else:
                    outv[x,y] = f[x,y] - f[x, y-1]
            else:
                if y == outv.shape[0]-1:
                    outv[x,y] = f[x,y] - f[x, y-1]
                elif x == 0:
                    outv[x,y] = f[x,y+1] - f[x,y]
                else:
                    outv[x,y] = f[x,y+1] - f[x, y-1]
    return [outu, outv]


def upwind_boundary_gradient2d(f):
    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'
    outu = np.zeros_like(f).astype(otype)
    outv = np.copy(outu)
    zeros = np.copy(outu[0,:])
    for i, x in enumerate(f):
        i_plus = (i+1) % f.shape[0]
        outu[i,:] = np.maximum(np.maximum(f[i-1,:] - f[i,:], f[i_plus,:] - f[i,:]), zeros)
    for j, y in enumerate(np.transpose(f)):
        j_plus = (j+1) % f.shape[1]
        outv[:,j] = np.maximum(np.maximum(f[:,j-1] - f[:,j], f[:,j_plus] - f[:,j]), zeros)
    return [outu, outv]

def lsm_grad_magnitude(f, plus=True):
    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'
    outu = np.zeros_like(f).astype(otype)
    outv = np.copy(outu)
    zeros = np.copy(outu[0,:])
    for i, x in enumerate(f):
        i_plus = (i+1) % f.shape[0]
        if plus:
            outu[i,:] = np.square(np.maximum(f[i-1,:] - f[i,:], zeros)) + \
                        np.square(np.minimum(f[i_plus,:] - f[i,:], zeros))
        else:
            outu[i,:] = np.square(np.maximum(f[i_plus,:] - f[i,:], zeros)) + \
                        np.square(np.minimum(f[i-1,:] - f[i,:], zeros))
    for j, y in enumerate(np.transpose(f)):
        j_plus = (j+1) % f.shape[1]
        if plus:
            outv[:,j] = np.square(np.maximum(f[:,j-1] - f[:,j], zeros)) + \
                        np.square(np.minimum(f[:,j_plus] - f[:,j], zeros))
        else:
            outv[:,j] = np.square(np.maximum(f[:,j_plus] - f[:,j], zeros)) + \
                        np.square(np.minimum(f[:,j-1] - f[:,j], zeros))
    return np.sqrt(outu + outv)  

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

arr = np.vander(np.array(xrange(4)))
# U, V = gradient(arr)
# print arr
# print U
# print V
# uu, uv = upwind_gradient2d(arr, np.ones((4,4)))
# print uu
# print uv
print gradient_magnitude(*np.gradient(arr))
print lsm_grad_magnitude(arr, True)
print lsm_grad_magnitude(arr, False)
