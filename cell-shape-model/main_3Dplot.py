import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate

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

params = parameters.Params()

shape_def = shape_definition.Shape_definition(params)

# phi = np.pi / 3.0
# phi = np.pi / 3.0

spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,
                                                           [params.r_base * (1.0 - np.cos(params.start_phi)),
                                                            params.r_base * np.sin(params.start_phi)],
                                                           params.step_size_natural,
                                                           params.smoothness_natural)

# spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,
#                                                            [params.r_base - params.R_base * np.cos(params.start_phi),
#                                                             params.R_base * np.sin(params.start_phi)], 0.001, params.smoothness_natural)

spline_relaxed = spline_interpolation.Spline_Interpolation(shape_def.pointsRelaxedShape,
                                                           [params.r_base - params.R_base * np.cos(params.start_phi),
                                                            params.R_base * np.sin(params.start_phi)],
                                                           params.step_size_relaxed,
                                                           params.smoothness_relaxed)
# spline_relaxed = spline_interpolation.Spline_Interpolation(shape_def.pointsRelaxedShape,[7.0,1.0],0.0001,params.smoothness_relaxed)


''' Show Spline Interploation'''

x1 = spline_natural.pointsShape[:, 0]
y1 = spline_natural.pointsShape[:, 1]

x_tip = max(x1) - spline_natural.arclength_to_x(spline_natural.total_arclength - params.s_tip)
x_neck = max(x1) - spline_natural.arclength_to_x(spline_natural.total_arclength - params.s_neck)
x_base = max(x1) - spline_natural.arclength_to_x(spline_natural.total_arclength - params.s_base)

print "s_tip --> x_tip"
print str(params.s_tip) + "-->" + str(x_tip)
print "s_neck --> x_neck"
print str(params.s_neck) + "-->" + str(x_neck)
print "s_base --> x_base"
print str(params.s_base) + "-->" + str(x_base)

fig = plt.figure(3, figsize=(9, 3), dpi=300)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax.set_xlim(min(x1) - 0.5, max(x1) + 0.5)
# ax.set_ylim(min(y1) - 0.5, max(y1) + 0.5)
ax.set_xlim(-0.5, max(x1) + 0.5)
ax.set_ylim(0.0, max(y1) + 0.5)

# ax.plot(x1, y1, 'x')
ax.plot(spline_natural.splineShape[0], spline_natural.splineShape[1], 'k')
ax.axvline(x_tip, color='k', linestyle='--')
ax.axvline(x_neck, color='k', linestyle='--')
ax.axvline(x_base, color='k', linestyle='--')

out = np.array(spline_natural.splineShape_s)
x2 = out[:, 0]
y2 = out[:, 1]
# ax.plot(x2, y2, color='r')
# ax.annotate("Start", xy=(2, 2), xytext=(x2[0], y2[0]))

x1 = spline_relaxed.pointsShape[:, 0]
y1 = spline_relaxed.pointsShape[:, 1]

# fig = plt.figure(3, figsize=(9, 3), dpi=300)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# ax.plot(x1, y1, 'x')
ax.plot(spline_relaxed.splineShape[0], spline_relaxed.splineShape[1], 'k--')
out = np.array(spline_relaxed.splineShape_s)
x2 = out[:, 0]
y2 = out[:, 1]
# ax.plot(x2, y2, '--')
# ax.annotate("Start", xy=(2, 2), xytext=(x2[0], y2[0]))



plt.title('Points defining the natural cell shape')
plt.savefig('Spline_Interpolation.eps', format="eps")
plt.savefig('Spline_Interpolation.pdf', format="pdf")
plt.show()

# spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,[params.L-0.2,0.2],0.001, params. smoothness_natural)

stress = stresses.Stresses(spline_natural, params)

''' Plot Stresses '''

total_arclength = stress.sigma_m[-1][2] + params.r_base * params.start_phi
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
    # s.append(stress.sigma_m[-1][2] - stress.sigma_theta[i][2] - params.epsilon)
    s.append(stress.sigma_m[-1][2] - stress.sigma_theta[i][2])

fig_stresses = plt.figure(3, figsize=(9, 3), dpi=300)

ax_stresses = fig_stresses.add_axes([0.1, 0.1, 0.8, 0.8])

# ax_strains.plot(s, y_m, '-', s, y_theta, '-')

ax_stresses.plot(s, sigma_m, '-', label=r'$\sigma_{m}$')
ax_stresses.plot(s, sigma_theta, '-', label=r'$\sigma_{\theta}$')
ax_stresses.plot(s, sigma_VM, '-', label=r'$\sigma_{VM}$')
legend = ax_stresses.legend(loc='upper right', shadow=True)

ax_stresses.tick_params(length=4, width=1.0, size=3, labelsize=8)
xtick_locs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
xtick_lbls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
plt.xticks(xtick_locs, xtick_lbls)

ax_stresses.set_ylim(-0.4, 4.6)
ax_stresses.set_xlim(params.epsilon, total_arclength)

ax_stresses.axvline(params.s_tip, color='k', linestyle='--')
ax_stresses.axvline(params.s_neck, color='k', linestyle='--')
ax_stresses.axvline(params.s_base, color='k', linestyle='--')

plt.title('Meridional and Circumferential Stress', fontsize=12)
plt.savefig('meridional_n_circumferential_stress.eps', format="eps")
plt.savefig('meridional_n_circumferential_stress.pdf', format="pdf")
plt.show()


''' 3D Plot Stresses'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

out = np.array(spline_natural.splineShape_s)
shape_x = out[:, 0]
shape_y = out[:, 1]
shape_s = out[:, 2]

print "len shape: " + str(len(shape_y))
print "len stress: " + str(len(sigma_VM))

#del sigma_VM[-1]

sigma_VM1 = []
shape_x1 = []
shape_y1 = []
steps = 100

for i in range(0,len(shape_x)/steps):
  sigma_VM1.append(sigma_VM[steps*i])
  shape_x1.append(shape_x[steps*i])
  shape_y1.append(shape_y[steps*i])

print "len shape: " + str(len(shape_y1))
print "len stress: " + str(len(sigma_VM1))

theta_grid = np.linspace(0, 2 * np.pi, 100)
r_points, theta_points = np.meshgrid(shape_y1, theta_grid)

data, theta_points = np.meshgrid(sigma_VM1, theta_grid)

x_points, y_points = r_points * np.cos(theta_points), r_points * np.sin(theta_points)

ax.plot_surface(x_points, y_points, shape_x1, rstride=1, cstride=1, cmap=cm.YlGnBu_r)

sigma_range = max(sigma_VM) - min(sigma_VM)

N = data / 3.0

# print N
# print cm.jet(N)

ax.plot_surface(x_points, y_points, shape_x1, rstride=1, cstride=1, facecolors=cm.coolwarm(N), linewidth=1,
                        antialiased=False, shade=False)

# ax.set_zlim3d(0, cell_radius + shmoo_length)
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.set_zlabel(r'$z$')

ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines() + a.get_ticklabels():
	t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)

m = cm.ScalarMappable(cmap=cm.coolwarm)
m.set_array(N)
m.set_clim(0.0, 3.0)
plt.colorbar(m)
plt.savefig("Stress.tif")
plt.savefig("Stress.pdf", format="pdf")
plt.show()


''' Calculate Strains & Elasticty '''

# phi = np.pi / 2.0
# phi = np.pi / 3.0

spline_natural = spline_interpolation.Spline_Interpolation(shape_def.pointsNaturalShape,
                                                           [params.r_base * (1.0 - np.cos(params.start_phi)),
                                                            params.r_base * np.sin(params.start_phi)],
                                                           params.step_size_natural,
                                                           params.smoothness_natural)
spline_relaxed = spline_interpolation.Spline_Interpolation(shape_def.pointsRelaxedShape,
                                                           [params.r_base - params.R_base * np.cos(params.start_phi),
                                                            params.R_base * np.sin(params.start_phi)],
                                                           params.step_size_relaxed,
                                                           params.smoothness_relaxed)

''' Show Spline Interploation'''

x1 = spline_natural.pointsShape[:, 0]
y1 = spline_natural.pointsShape[:, 1]

fig = plt.figure(3, figsize=(9, 3), dpi=300)

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
# plt.savefig('Spline_Interpolation.eps', format="eps")
# plt.savefig('Spline_Interpolation.pdf', format="pdf")
plt.show()

strain = strains.Strain(spline_natural, spline_relaxed, params)
s_strain = [total_arclength]  # ,total_arclength - 2.0*params.r_base * params.start_phi/3.0,]
strain_temp = params.r_base / params.R_base - 1.0
strain_m = [strain_temp]
strain_theta = [strain_temp]
strain_vol = [2.0 * strain_temp]
E_temp = (1.0 - params.nu ** 2) * (1.0 - params.nu) * (0.5 * params.P * params.r_base / params.h) * params.R_base / (
    params.r_base - params.R_base)
E_theta = [E_temp]

for i in range(0, len(strain.strain_m)):
    if (strain.strain_m[i][0] < total_arclength - params.s_tip - params.r_base * params.start_phi):
        s_strain.append(total_arclength - strain.strain_m[i][0] - params.r_base * params.start_phi)
        strain_m.append(strain.strain_m[i][1])
        strain_theta.append(strain.strain_theta[i][1])
        strain_vol.append(strain_m[-1] + strain_theta[-1])
        E_theta.append(strain.E_theta[i][1])

# s_strain.append(total_arclength)
# strain_m.append(params.r_base/params.R_base - 1.0)
# strain_theta.append(params.r_base/params.R_base - 1.0)
# strain_vol.append(2.0*params.r_base/params.R_base - 2.0)

fig_strains = plt.figure(3, figsize=(9, 3), dpi=300)

ax_strains = fig_strains.add_axes([0.1, 0.1, 0.8, 0.8])

# ax_strains.plot(s, y_m, '-', s, y_theta, '-')

ax_strains.plot(s_strain, strain_theta, '-', label=r'$\epsilon_{\theta}$')
ax_strains.plot(s_strain, strain_m, '-', label=r'$\epsilon_{m}$')
ax_strains.plot(s_strain, strain_vol, '-', label=r'$\epsilon_{vol}$')
# legend = ax_strains.legend(loc='upper right', shadow=True)
# ax_strains.plot(s_strain, strain_theta, '-', legend ='epsilon_theta')
# ax_strains.plot(s_strain, strain_m, '-', legend ='epsilon_m')
# ax_strains.plot(s_strain, strain_vol, '-', legend ='epsilon_vol')
legend = ax_strains.legend(loc='upper right')

ax_strains.set_ylim(-0.2, 1.8)
ax_strains.set_xlim(params.s_tip, total_arclength)

# ax_strains.plot(s_array[1], strain_m_array[1], '--', s_array[1], strain_theta_array[1], '-')
# ax_strains.plot(s_array[2], strain_m_array[2], '--', s_array[2], strain_theta_array[2], '-')

# ax_strains.tick_params(length=4, width=1.0, size=3, labelsize=8)
# xtick_locs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
# xtick_lbls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
# plt.xticks(xtick_locs, xtick_lbls)

# ax_strains.axvline(s_tip, color = 'k', linestyle = '--')
ax_strains.axvline(params.s_neck, color='k', linestyle='--')
ax_strains.axvline(params.s_base, color='k', linestyle='--')

plt.title('Meridional and Circumferential Strain', fontsize=12)
plt.savefig('meridional_n_circumferential_strain.eps', format="eps")
plt.savefig('meridional_n_circumferential_strain.pdf', format="pdf")
plt.show()

fig_strains = plt.figure(3, figsize=(9, 3), dpi=300)

ax_stresses = fig_strains.add_axes([0.1, 0.1, 0.8, 0.8])
ax_strains = ax_stresses.twinx()

# ax_strains.plot(s, y_m, '-', s, y_theta, '-')

# ax_strains.plot(s_strain, strain_theta, '-')
# ax_strains.plot(s_strain, strain_m, '-')
ax_strains.plot(s_strain, strain_vol, '-', label=r'$\epsilon_{vol}$')
ax_stresses.plot(s_strain, E_theta, '-', label='$E$')
ax_stresses.plot(s, sigma_VM, '-', label=r'$\sigma_{VM}$')
legend = ax_strains.legend(loc='upper right', shadow=True)
legend = ax_stresses.legend(loc='lower right', shadow=True)
ax_stresses.set_ylim(-0.2, 3.8)
ax_stresses.set_xlim(params.s_tip, total_arclength)
ax_strains.set_ylim(-0.4, 1.6)

# ax_stresses.axvline(s_tip, color = 'k', linestyle = '--')
ax_stresses.axvline(params.s_neck, color='k', linestyle='--')
ax_stresses.axvline(params.s_base, color='k', linestyle='--')

# ax_strains.plot(s_array[1], strain_m_array[1], '--', s_array[1], strain_theta_array[1], '-')
# ax_strains.plot(s_array[2], strain_m_array[2], '--', s_array[2], strain_theta_array[2], '-')

# ax_strains.tick_params(length=4, width=1.0, size=3, labelsize=8)
# xtick_locs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
# xtick_lbls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
# plt.xticks(xtick_locs, xtick_lbls)

plt.title('Elasticity', fontsize=12)
plt.savefig('Elasticity.eps', format="eps")
plt.savefig('Elasticity.pdf', format="pdf")
plt.show()



''' 3D Plot Strains'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

out = np.array(spline_natural.splineShape_s)
shape_x = out[:, 0]
shape_y = out[:, 1]
shape_s = out[:, 2]

print "len shape: " + str(len(shape_y))
print "len strain: " + str(len(strain_vol))

del strain_vol[-1]

strain_vol1 = []
shape_x1 = []
shape_y1 = []
steps = 100

strain_max = 1.6
strain_max = 1.6

for i in range(0,len(strain_vol)/steps):
  strain_vol1.append(max(0,min(strain_vol[steps*i],strain_max)))
  shape_x1.append(shape_x[steps*i])
  shape_y1.append(shape_y[steps*i])



print "len shape: " + str(len(shape_y))
print "len strain: " + str(len(strain_vol))

theta_grid = np.linspace(0, 2 * np.pi, 100)
r_points, theta_points = np.meshgrid(shape_y1, theta_grid)

data, theta_points = np.meshgrid(strain_vol1, theta_grid)

x_points, y_points = r_points * np.cos(theta_points), r_points * np.sin(theta_points)

#ax.plot_surface(x_points, y_points, shape_x1, rstride=1, cstride=1, cmap=cm.YlGnBu_r)

#strain_range = max(strain_vol) - min(strain_vol)
#strain_range = max(strain_vol) - min(strain_vol)

N = data / 1.8

# print N
# print cm.jet(N)

ax.plot_surface(x_points, y_points, shape_x1, rstride=1, cstride=1, facecolors=cm.coolwarm(N), linewidth=1,
                        antialiased=False, shade=False)

# ax.set_zlim3d(0, cell_radius + shmoo_length)
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.set_zlabel(r'$z$')

ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines() + a.get_ticklabels():
	t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)

m = cm.ScalarMappable(cmap=cm.coolwarm)
m.set_array(N)
m.set_clim(0.0, 1.8)
plt.colorbar(m)
plt.savefig("Strain.tif")
plt.savefig("Strain.pdf", format="pdf")
plt.show()



''' 3D Plot Strains'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

out = np.array(spline_natural.splineShape_s)
shape_x = out[:, 0]
shape_y = out[:, 1]
shape_s = out[:, 2]

print "len shape: " + str(len(shape_y))
print "len strain: " + str(len(E_theta))


E_theta1 = []
shape_x1 = []
shape_y1 = []
steps = 100

E_max = 2.5
E_max = 2.5

for i in range(0,len(strain_vol)/steps):
  E_theta1.append(max(0.4,min(E_theta[steps*i],E_max)))
  shape_x1.append(shape_x[steps*i])
  shape_y1.append(shape_y[steps*i])



print "len shape: " + str(len(shape_y1))
print "len strain: " + str(len(E_theta1))

theta_grid = np.linspace(0, 2 * np.pi, 100)
r_points, theta_points = np.meshgrid(shape_y1, theta_grid)

data, theta_points = np.meshgrid(E_theta1, theta_grid)

x_points, y_points = r_points * np.cos(theta_points), r_points * np.sin(theta_points)

#ax.plot_surface(x_points, y_points, shape_x1, rstride=1, cstride=1, cmap=cm.YlGnBu_r)

#strain_range = max(strain_vol) - min(strain_vol)
#strain_range = max(strain_vol) - min(strain_vol)

N = data / 4.0

# print N
# print cm.jet(N)

ax.plot_surface(x_points, y_points, shape_x1, rstride=1, cstride=1, facecolors = cm.Greens_r(N), linewidth=1,
                        antialiased=False, shade=False)

# ax.set_zlim3d(0, cell_radius + shmoo_length)
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.set_zlabel(r'$z$')

ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines() + a.get_ticklabels():
	t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)

m = cm.ScalarMappable(cmap=cm.Greens_r)
m.set_array(N)
m.set_clim(0.0, 4.0)
plt.colorbar(m)
plt.savefig("Elasticity.tif")
plt.savefig("Elasticity.pdf", format="pdf")
plt.show()

