import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate

''' own classes '''

import shape_definition
import spline_interpolation
import stresses
import strains
import elasticity


class Params:
    ''' User parameters are defined here  '''

    def __init__(self):
        ''' Parameters Physics '''
        self.P = 0.2
        self.h = 0.12
        self.nu = 0.5

        ''' Parameters Shape '''
        self.R_base = 2.0
        self.r_base = 2.5
        self.R_shaft = 0.4
        self.r_shaft = 1.7
        self.r_tip = 0.7
        self.tipgrowth_radius = 0.2
        self.L = 8.0
        self.transistion_length = 2.2

        ''' output parameters '''
        self.interval = [0.1, 8.0]
        self.elasticityInterval = [0.5, 7.7]
        self.breakThreshold_epsilonTheta = 0.1
        self.breakThreshold_E = 0.5


params = Params()

''' Shape Definition '''

shape_def = shape_definition.Shape_definition(params)

#shape_def.show_pointsNaturalShape()
#shape_def.show_relaxedShape(params)
#shape_def.show_bothShapes(params)


''' Spline Interpolation '''

phi = np.pi/10.0

#spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,[params.r_base*np.cos(phi),params.r_base*np.sin(phi)],0.0001,0.01)
#spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,[params.r_base - params.R_base*np.cos(phi),params.R_base*np.sin(phi)],0.0001,0.01)
spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,[7.3,1.0],0.0001,0.02)

#spline_int.show_naturalshape()

#spline_relaxed = spline_interpolation.Spline_Interpolation(shape_def.pointsRelaxedShape,[params.r_base - params.R_base*np.cos(phi),params.R_base*np.sin(phi)],0.0001,0.01)
spline_relaxed = spline_interpolation.Spline_Interpolation(shape_def.pointsRelaxedShape,[7.0,1.0],0.0001,0.01)

#spline_int2.show_naturalshape()

''' Show Spline Interploation'''

x1 = spline_natural.pointsShape[:, 0]
y1 = spline_natural.pointsShape[:, 1]

fig = plt.figure(3, figsize=(9, 9), dpi=300)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_xlim(min(x1) - 0.5, max(x1) + 0.5)
ax.set_ylim(min(y1) - 0.5, max(y1) + 0.5)

ax.plot(x1, y1, 'x', spline_natural.splineShape[0], spline_natural.splineShape[1], 'b')
out = np.array(spline_natural.splineShape_s)
x2 = out[:, 0]
y2 = out[:, 1]
ax.plot(x2, y2, color='r')
ax.annotate("Start", xy=(2, 2), xytext=(x2[0], y2[0]))

x1 = spline_natural.pointsShape[:, 0]
y1 = spline_natural.pointsShape[:, 1]

fig = plt.figure(3, figsize=(9, 9), dpi=300)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.plot(x1, y1, 'x', spline_relaxed.splineShape[0], spline_relaxed.splineShape[1], 'b')
out = np.array(spline_relaxed.splineShape_s)
x2 = out[:, 0]
y2 = out[:, 1]
ax.plot(x2, y2, color='r')
ax.annotate("Start", xy=(2, 2), xytext=(x2[0], y2[0]))

plt.title('Points defining the natural cell shape')
plt.savefig('Spline_Interpolation.eps', format="eps")
plt.savefig('Spline_Interpolation.pdf', format="pdf")
plt.show()



''' Stress Calculations '''

stress = stresses.Stresses(spline_natural,params)

# stress.calculate_maximalStress(True)
# stress.calculate_vonMisesStress(True)


''' Strain Calculations '''

strain = strains.Strain(spline_natural, spline_relaxed, params)


''' Plot Elasticity '''

print "Elasticity"
print strain.E_m

out = np.array(strain.E_m)
s = out[:, 0]
y_m = out[:, 1]


tck, u = interpolate.splprep([s, y_m], k=5, s=0.2)
u = np.arange(0, 1.01, 0.01)
youngs_modulus = interpolate.splev(u, tck)

fig = plt.figure(3, figsize=(9, 9), dpi=300)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

#ax.plot(s, y_m, 'x')
ax.plot(youngs_modulus[0], youngs_modulus[1], '-')
ax.set_xlim(0.0,8.0)
ax.set_ylim(0.0,5.0)

plt.title('E interpolated')
plt.savefig('E_interpolated.eps', format="eps")
plt.savefig('E_interpolated.pdf', format="pdf")
plt.show()


