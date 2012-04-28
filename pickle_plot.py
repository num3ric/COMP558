from __future__ import division
import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import pickle

fig = plt.figure()
def load_p(number):
    with open(str(number)+'p.pickle') as f:
        fig = plt.figure()
        P = pickle.load(f)
        im = plt.imshow(P, cmap='jet')
        plt.contour(P, levels=[0])
        fig.colorbar(im)
        plt.draw()


saved_frames = [0, 84, 220, 280]

for frame in saved_frames:
    load_p(frame)

plt.show()