import numpy as np
from mayavi.mlab import *

x = [232,300,200,250,260,270,265,267,255]
y = [232,300,200,250,260,270,265,267,255]
z=[17.02,20.83,21.25,15.0,10.58,11.11,10.38,11.11,11.76]
s=10
points3d(x,y,z,s, colormap='copper',scale_factor=.25)