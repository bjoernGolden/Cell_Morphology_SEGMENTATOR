import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate
import csv

# ''' For Tex Usage'''
from matplotlib import rc

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

''' own classes '''

import shape_definition
import spline_interpolation
import stresses
import strains
import elasticity
import parameters


# class Params:
#     ''' User parameters are defined here  '''
#
#     def __init__(self):
#         ''' Parameters Physics '''
#         self.P = 0.2
#         self.h = 0.12
#         self.nu = 0.5
#
#         ''' Parameters Shape '''
#         self.R_base = 1.9
#         self.r_base = 2.5
#         self.R_shaft = 0.6
#         self.r_shaft = 1.7
#         self.r_tip = 0.7
#         self.tipgrowth_radius = 0.2
#         self.L = 8.0
#         self.transistion_length = 2.2
#
#         ''' Definition of Regions '''
#
#         self.s_tip = 1.5
#         self.s_neck = 4.0
#         self.s_base = 6.0
#
#         ''' Output Parameters '''
#         self.interval = [0.1, 8.0]
#         self.elasticityInterval = [0.5, 7.7]
#         self.breakThreshold_epsilonTheta = 0.1
#         self.breakThreshold_E = 0.5
#         self.epsilon = 0.2
#
#     def set_Rshaft(self, R):
#         self.R_shaft = R


R = [0.55, 0.6, 0.65]

file_path = 'natural_shape_points.csv'
output_file = open(file_path, 'w')
data = csv.writer(output_file)
data.writerow(['x', 'y'])

# data.writerow([2.0, 1.0])
# del data
# output_file.close()
# del output_file


# # csv.open('natural_shape_points.csv', 'w')
# file_path = 'natural_shape_points.csv'
# with open(file_path, "w") as file:
#     file.writerow(['x', 'y'])
#     file.writerow([1.0, 2.0])
#     file.close()

# out = csv.writer(open('natural_shape_points.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)
# out.writerow(['x','y'])
# out.writerow([1.0,2.0])
# out.close()
# writer = csv.DictWriter(out, fieldnames=fieldnames)

# writer.writeheader()
# writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
# writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
# writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

params = parameters.Params()

''' Shape Definition '''

pointsNaturalShape = []

circle_sections = 50

for i in range(0, circle_sections):
    theta = (np.pi / (1.7 * circle_sections)) * i
    # pointsNaturalShape.append([params.r_base * (1.0 - np.cos(np.pi / 1.7 - theta)), params.r_base * np.sin(np.pi / 1.7 - theta)])
    pointsNaturalShape.append([params.r_base * (1.0 - np.cos(theta)), params.r_base * np.sin(theta)])

## Transition region


pointsNaturalShape.append([params.r_base + 1.4, params.r_base - 0.3])

## Construct Shaft

sections = 3

shaft_end = params.L - params.r_tip

shaft_start = params.r_base + params.transistion_length

for i in range(0, sections):
    lamda = (1.0 / sections) * i
    pointsNaturalShape.append([(1.0 - lamda) * shaft_start + lamda * shaft_end,
                               (1.0 - lamda) * params.r_shaft + lamda * params.r_tip])

pointsNaturalShape[-2][1] -= 0.05

## Construct tip with radius r_tip

circle_sections = 5

# for i in range(2, circle_sections):
##theta = (np.pi / (2.0 * circle_sections)) * i
# theta = (np.pi / (2.0 * circle_sections)) * i
##pointsNaturalShape.append([params.L - params.r_tip * (1.0 - np.sin(theta)), params.r_tip * np.cos(theta)])
# pointsNaturalShape.append([params.L - params.r_base * (1.0 - np.sin(theta)), params.r_base * np.cos(theta)])

theta = (np.pi / (2.0 * circle_sections)) * 2
pointsNaturalShape.append(
    # [params.L - params.r_tip * (1.0 - np.sin(theta)), params.r_tip * np.cos(theta) + 0.05])
    [params.L - params.r_tip * (1.0 - np.sin(theta)), params.r_tip * np.cos(theta)])

theta = (np.pi / (2.0 * circle_sections)) * 3
pointsNaturalShape.append(
    [params.L - params.r_tip * (1.0 - np.sin(theta)) - 0.15, params.r_tip * np.cos(theta)])

theta = (np.pi / (2.0 * circle_sections)) * 4
pointsNaturalShape.append(
    [params.L - params.r_tip * (1.0 - np.sin(theta)) - 0.21, params.r_tip * np.cos(theta)])

for i in range(0, len(pointsNaturalShape)):
    data.writerow([pointsNaturalShape[i][0], pointsNaturalShape[i][1]])

# data.writerow([2.0, 1.0])
del data
output_file.close()
del output_file

shape_def = shape_definition.Shape_definition(params)

phi = np.pi / 3.0

spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,
                                                           [params.r_base * (1.0 - np.cos(phi)),
                                                            params.r_base * np.sin(phi)], params.step_size_natural,
                                                           params.smoothness_relaxed)

# spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,
#                                                            [params.r_base - params.R_base * np.cos(phi),
#                                                             params.R_base * np.sin(phi)], 0.001, 0.1)

spline_relaxed = spline_interpolation.Spline_Interpolation(shape_def.pointsRelaxedShape,
                                                           [params.r_base - params.R_base * np.cos(phi),
                                                            params.R_base * np.sin(phi)], params.step_size_relaxed,
                                                           params.smoothness_relaxed)
# spline_relaxed = spline_interpolation.Spline_Interpolation(shape_def.pointsRelaxedShape,[7.0,1.0],0.0001,0.0)


''' Show Spline Interploation'''

x1 = spline_natural.pointsShape[:, 0]
y1 = spline_natural.pointsShape[:, 1]

fig = plt.figure(3, figsize=(9, 9), dpi=300)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax.set_xlim(min(x1) - 0.5, max(x1) + 0.5)
# ax.set_ylim(min(y1) - 0.5, max(y1) + 0.5)
ax.set_xlim(0.0, max(x1) + 0.5)
ax.set_ylim(0.0, max(y1) + 0.5)

ax.plot(x1, y1, 'x', spline_natural.splineShape[0], spline_natural.splineShape[1], 'b')
out = np.array(spline_natural.splineShape_s)
x2 = out[:, 0]
y2 = out[:, 1]
ax.plot(x2, y2, color='r')
ax.annotate("Start", xy=(2, 2), xytext=(x2[0], y2[0]))

x1 = spline_relaxed.pointsShape[:, 0]
y1 = spline_relaxed.pointsShape[:, 1]

fig = plt.figure(3, figsize=(9, 9), dpi=300)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.plot(x1, y1, 'x', spline_relaxed.splineShape[0], spline_relaxed.splineShape[1], '--')
out = np.array(spline_relaxed.splineShape_s)
x2 = out[:, 0]
y2 = out[:, 1]
ax.plot(x2, y2, '--')
ax.annotate("Start", xy=(2, 2), xytext=(x2[0], y2[0]))

plt.title('Points defining the natural cell shape')
plt.savefig('test_shape.eps', format="eps")
plt.savefig('test_shape.pdf', format="pdf")
plt.show()

# spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,[params.L-0.2,0.2],0.001,0.02)

stress = stresses.Stresses(spline_natural, params)

''' Plot Stresses '''

total_arclength = stress.sigma_m[-1][2] + params.r_base * phi
s = [total_arclength]
temp = params.P * params.r_base / (2.0 * params.h)
sigma_m = [temp]
sigma_theta = [temp]

sigma_VM = [temp]

for i in range(0, len(stress.sigma_theta)):
    sigma_theta.append(stress.sigma_theta[i][1])
    sigma_m.append(stress.sigma_m[i][1])
    vm = np.sqrt(sigma_m[-1] ** 2 + sigma_theta[-1] ** 2 - sigma_m[-1] * sigma_theta[-1])
    sigma_VM.append(vm)
    s.append(stress.sigma_m[-1][2] - stress.sigma_theta[i][2] - params.epsilon)

fig_stresses = plt.figure(3, figsize=(9, 5), dpi=300)

ax_stresses = fig_stresses.add_axes([0.1, 0.1, 0.8, 0.8])

# ax_strains.plot(s, y_m, '-', s, y_theta, '-')

ax_stresses.plot(s, sigma_m, '-', label=r'$\sigma_{m}$')
ax_stresses.plot(s, sigma_theta, '-', label=r'$\sigma_{\theta}$')
ax_stresses.plot(s, sigma_VM, '-', label=r'$\sigma_{VM}$')
legend = ax_stresses.legend(loc='upper right', shadow=True)

ax_stresses.tick_params(length=4, width=1.0, size=3, labelsize=8)
xtick_locs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
xtick_lbls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
plt.xticks(xtick_locs, xtick_lbls)

ax_stresses.set_ylim(-0.4, 4.6)
ax_stresses.set_xlim(0.0, total_arclength)

plt.title('Meridional and Circumferential Stress', fontsize=12)
plt.savefig('test_stress.eps', format="eps")
plt.savefig('test_stress.pdf', format="pdf")
plt.show()

phi = np.pi / 2.0
