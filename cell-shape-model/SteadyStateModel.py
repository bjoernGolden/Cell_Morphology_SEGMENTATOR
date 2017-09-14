import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate


''' own classes '''

import shape_definition



class Params:
  ''' User parameters are defined here  '''
  def __init__(self):
    
    ''' Parameters Physics ''' 
    self.P = 0.1
    self.h = 0.1
    self.nu = 0.5
    
    ''' Parameters Shape '''
    self.R_base = 2.0
    self.r_base = 2.5
    self.R_shaft = 0.4
    self.r_shaft = 1.7
    self.r_tip = 0.7
    self.tipgrowth_radius = 0.2
    self.L = 8.0
    self.transistion_length = 1.5


params = Params()

print "R_base" + str(params.R_base)

P = 0.1
h = 0.1
nu = 0.5


  

R_base = 2.0
r_base = 2.5
R_shaft = 0.4
r_shaft = 1.7
r_tip = 0.7
tipgrowth_radius = 0.2
L = 8.0
transistion_length = 1.5

''' Construct geometry '''

points = []  # Points for spline fitting

# Construct tip with radius r_tip

circle_sections = 5

for i in range(0, circle_sections):
    theta = (np.pi / (3.0 * circle_sections)) * i
    points.append([r_tip * (1.0 - np.cos(theta)), r_tip * np.sin(theta)])

# Construct Shaft

sections = 3
shaft_length = L - r_tip - r_base - transistion_length

for i in range(0, sections):
    lam = (1.0 / sections) * i
    points.append([(1.0 - lam) * r_tip + lam * shaft_length, (1.0 - lam) * r_tip + lam * r_shaft])

# Transition region


points.append([L - r_base - 1.4, r_base - 0.1])


# This point works good
# points.append([L - r_base  - 1.4 ,r_base-0.1])


# Construct Base


circle_sections = 10

for i in range(0, circle_sections):
    theta = (np.pi / (1.7 * circle_sections)) * i
    points.append([L - r_base * (1.0 - np.cos(np.pi / 1.7 - theta)), r_base * np.sin(np.pi / 1.7 - theta)])




# for i in range(0,circle_sections):
# theta = (np.pi/(1.7*circle_sections))*i
# points.append([L - r_base*(1.0 - np.cos(np.pi/1.7 - theta)),r_base*np.sin(np.pi/1.7 - theta)])


''' Plot points '''

temp_points = np.array(points)
x = temp_points[:, 0]
y = temp_points[:, 1]

plt.figure()
plt.plot(x, y, 'b')
plt.title('Points Defining the Geometry')
plt.show()

x_pos = []

for i in range(0, len(points)):
    x_pos.append(points[i][0])

for i in range(0, len(points)):
    points[i][0] = L - x_pos[i]

''' Reverse points --> First point starts now at the base '''

points.reverse()

print "after reverse:"
print points

temp_points = np.array(points)
x = temp_points[:, 0]
y = temp_points[:, 1]

plt.figure()
plt.plot(x, y, 'b')
# Plot start and end points
plt.plot([points[0][0]], [points[0][1]], 'o', [points[-1][0]], [points[-1][1]], 'x')

plt.title('Reversed Geometry')
plt.show()

# Construct lower part

temp = len(points) - 1
for i in range(1, len(points)):
    x = points[temp - i][0]
    y = -points[temp - i][1]
    points.append([x, y])


# points.append([points[0][0], -points[0][1]])
points.append([0, 0.0])
points.insert(0, [0, 0.0])


def R_fct(x):
    R = R_shaft

    if (x <= R_base + r_base):
        temp = np.sqrt(R_base ** 2 - (x - r_base) ** 2)
        R = temp
    if (x <= r_base - R_base):
        R = R_shaft

    return R

# def R_fct(x):
# R = tipgrowth_radius

# if( (x > 2*R_base) ):
# R = R_shaft
# else:
# temp = np.sqrt(R_base**2 - (x-R_base)**2)
# R = temp

# return R


# Interpolate Splines

points = np.array(points)
x = points[:, 0]
y = points[:, 1]

tck, u = interpolate.splprep([x, y], k=5, s=0.1)
# unew = np.arange(0,1.01,0.01)
unew = np.arange(0, 1.01, 0.005)
out = interpolate.splev(unew, tck)

x_old = L - r_base + R_base * np.sin(2 * np.pi * unew)
# for i in range(0,len(x_old)):
# x_old[i] += L - r_base
y_old = R_base * np.cos(2 * np.pi * unew)


# Plot Spline

plt.figure()
plt.plot(x, y, 'x', out[0], out[1], x_old, y_old, 'b')
# plt.legend(['Linear','Cubic Spline', 'True'])


# plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Spline of parametrically-defined curve')
plt.show()

''' Calcualte Derivatives '''

out_dev = interpolate.splev(unew, tck, der=1)
out_dev2 = interpolate.splev(unew, tck, der=2)

''' cut lower half '''

outx_upper = []
outy_upper = []
devx_upper = []
devy_upper = []
dev2x_upper = []
dev2y_upper = []
R_ = []

epsilon_l = 0.01
# epsilon_r = r_base - R_base
epsilon_r = 0.0

for i in range(0, len(out[0]) - 1):
    if ((out[1][i] > tipgrowth_radius + epsilon_l) & (out[0][i] < L - epsilon_r)):
        # if( out[1][i]>= 0.0):
        outx_upper.append(out[0][i])
        outy_upper.append(out[1][i])
        devx_upper.append(out_dev[0][i])
        devy_upper.append(out_dev[1][i])
        dev2x_upper.append(out_dev2[0][i])
        dev2y_upper.append(out_dev2[1][i])
        R_.append(R_fct(out[0][i]))

fig = plt.figure(1, figsize=(max(x), max(y)), dpi=100)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax2 = ax.twinx()

# Plot actual cell form and relaxed cell form

ax.plot(outx_upper, outy_upper, 'b', outx_upper, R_)
# ax2.plot(outx_upper,devy_upper,'--')
# ax.legend(['Form','dx/du'])
# plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Spline of parametrically-defined curve')
plt.show()

''' Calculate meridional curvature '''

# Cut first half of outx_upper away

kappa_m = []
x_points = []
start_index = 10

for i in range(start_index, len(outx_upper)):
    sign = 1.0
    if (devx_upper[i] < 0.0):
        sign = -1.0
    nominator = - sign * (dev2y_upper[i] * devx_upper[i] - dev2x_upper[i] * devy_upper[i])
    denominator = np.sqrt(devx_upper[i] ** 2 + devy_upper[i] ** 2) ** 3
    temp = nominator / denominator
    kappa_m.append(temp)
    x_points.append(outx_upper[i])

fig = plt.figure(1, figsize=(max(x), max(y)), dpi=100)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = ax.twinx()

ax.plot(x_points, kappa_m, 'b')
ax2.plot(outx_upper, outy_upper, '--')
# ax.legend(['Form','dx/du'])
# plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Meridional Curvature')
plt.show()

''' Calculate circumferential curvature '''

kappa_theta = []

for i in range(start_index, len(outx_upper)):
    nominator = devx_upper[i]
    denominator = np.absolute(outy_upper[i]) * np.sqrt(devx_upper[i] ** 2 + devy_upper[i] ** 2)
    temp = nominator / denominator
    kappa_theta.append(temp)

quotient = []
for i in range(0, len(kappa_m)):
    quotient.append(kappa_m[i] / kappa_theta[i])

fig = plt.figure(1, figsize=(max(x), max(y)), dpi=100)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = ax.twinx()

ax.plot(x_points, kappa_theta, 'b')
ax2.plot(outx_upper, outy_upper, '--')
# ax.legend(['Form','dx/du'])
# plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Circumferential curvature')
plt.show()

''' calculate arclength '''

print "Calculate arclength"

ds = []
dS = []
xs = [outx_upper[start_index]]
ys = [outy_upper[start_index]]

phi = np.arctan2(outy_upper[start_index], r_base - outx_upper[start_index])
print phi

xS = [r_base - R_base * np.cos(phi)]
yS = [R_base * np.sin(phi)]
# xS = []
# yS = []



''' initialize arclength '''

arclength_s = [r_base * phi]
arclength_relaxed = [0.0]
arclength_S = [R_base * phi]

''' calculate start point & index for the relaxed curve  '''
start_x_relaxed = R_base * outx_upper[start_index] / r_base
start_y_relaxed = R_base * outy_upper[start_index] / r_base

start_i_relaxed = 0

dist = L
for i in range(0, len(outx_upper)):
    temp = np.sqrt((outx_upper[i] - start_x_relaxed) ** 2 + (R_[i] - start_y_relaxed) ** 2)
    if (temp < dist):
        dist = temp
        start_i_relaxed = i

ds_indizes = [0]
remainder = 0.0
dS_sum = phi * R_base
ds_sum = phi * r_base
S = 0.0
# index_ds_to_dS = [0]




# for i in range(1,len(outx_upper)):
# dS = np.sqrt((outx_upper[len(outx_upper)-i] - outx_upper[len(outx_upper) - i - 1])**2 + (R_[len(outx_upper)-i] - R_[len(outx_upper) -i-1])**2)
# S += dS
# arclength_relaxed.append(S)

for i in range(0, len(outx_upper) - 1):
    dS = np.sqrt((outx_upper[i] - outx_upper[i + 1]) ** 2 + (R_[i] - R_[i + 1]) ** 2)
    S += dS
    arclength_relaxed.append(S)


def index_ds_to_dS(x):
    index = 0
    for i in range(0, len(outx_upper) - 1):
        if (arclength_relaxed[i] <= x) & (x <= arclength_relaxed[i + 1]):
            index = i
    return index


theta = np.arcsin(R_shaft / R_base)
print "-----"
print "theta: " + str(theta)
print "-----"


def arclength_S_to_yS(arc_S):
    transition_S = R_base * (np.pi - theta)
    yS_ = R_shaft
    if (arc_S < transition_S):
        phi = arc_S / R_base
        yS_ = R_base * np.sin(phi)
    return yS_


def arclength_S_to_xS(arc_S):
    transition_S = R_base * (np.pi - theta)
    xS_ = 0.0
    if (arc_S < transition_S):
        phi = arc_S / R_base
        xS_ = r_base - R_base * np.cos(phi)
    else:
        xS_ = r_base + R_base * np.cos(theta) + arc_S - transition_S

    return xS_

# def arclength_S_to_Rx(arc_S):
# transition_S = R_base*(np.pi - theta)
# R_x = R_shaft
# if (arc_S < transition_S):
# phi = arc_S/R_base
# R_x = R_base*np.sin(phi)
# return R_x

for i in range(start_index, len(outx_upper) - 1):
    ds_value = np.sqrt((outx_upper[i] - outx_upper[i + 1]) ** 2 + (outy_upper[i] - outy_upper[i + 1]) ** 2)
    ds_sum += ds_value
    arclength_s.append(ds_sum)

    # index =  index_ds_to_dS(arclength_S[-1])


    # if (index < len(outx_upper)-2):
    # factor = (outy_upper[i]/R_[index] - 1.0)
    # else:
    # factor = outy_upper[i]/R_shaft - 1.0

    yS_ = arclength_S_to_yS(arclength_S[-1])
    xS_ = arclength_S_to_xS(arclength_S[-1])


    # scaling_factor = r_base/R_base
    # if ( (kappa_theta[i-start_index]-1.0/r_base)**2 + (kappa_m[i - start_index] - 1.0/r_base)**2 > 0.1 ):
    factor = (outy_upper[i] / yS_ - 1.0)
    nominator = (1.0 - 2.0 * nu) * kappa_theta[i - start_index] + nu * kappa_m[i - start_index]
    denominator = (2.0 - nu) * kappa_theta[i - start_index] - kappa_m[i - start_index]
    scaling_factor = factor * nominator / denominator + 1.0


    # scaling_factor *= 0.9
    ds.append(ds_value)

    dS_value0 = ds_value / scaling_factor
    # dS.append(dS_value0)
    dS_sum += dS_value0
    arclength_S.append(dS_sum)

    print "scaling_factor: " + str(scaling_factor)
    print "arclength_s: " + str(arclength_s[-1])
    print "arclength_S: " + str(arclength_S[-1])
    print "kappa_m: " + str(kappa_m[i - start_index])
    print "kappa_theta: " + str(kappa_theta[i - start_index])
    print "R: " + str(yS_)
    print "r: " + str(outy_upper[i])
    print "ratio: " + str(yS_ / outy_upper[i])

    # print "arclength_relaxed: " + str(arclength_relaxed[i])



    # while ( (arclength_relaxed[i] > arclength_S[i]) & (arclength_relaxed[i+1] < arclength_S[i])  ):
    # index += 1
    # print index
    # print (len(outx_upper)-1)
    # for j in range(0,len(outx_upper)-1):
    # if ( (arclength_relaxed[j] <= arclength_S[index]) & ( arclength_S[index] < arclength_relaxed[j+1]) ):
    # index = j
    # print index
    # print arclength_relaxed[index]
    # print arclength_S[index]
    # index += 1



    # if ( (index < len(outx_upper)-1) & (index > 0) ):
    # yS.append(R_[index])
    # xS.append(outx_upper[index])
    # print xS[-1]
    # print index
    # else:
    yS.append(yS_)
    xS.append(xS_)
    xs.append(outx_upper[i])
    ys.append(outy_upper[i])
    # ds_indizes.append(index)


# transition_index = 0

# for i in range(0,len(outx_upper)):
# if (S[i] > L - R_base):
# transition_index = i
# breakstart


# print xS
# print yS
# print R_

fig = plt.figure(1, figsize=(max(x), max(y)), dpi=100)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax2 = ax.twinx()

# Plot actual cell form and relaxed cell form

ax.plot(xs, ys, 'b', xS, yS, '--')

xs_mod0 = []
ys_mod0 = []

xS_mod0 = []
yS_mod0 = []

for i in range(0, (len(xs) - 1) / 5):
    xs_mod0.append(xs[i * 5])
    ys_mod0.append(ys[i * 5])
    yS_mod0.append(yS[i * 5])
    xS_mod0.append(xS[i * 5])

ax.plot(xs_mod0, ys_mod0, 'x', xS_mod0, yS_mod0, 'x', [r_base - R_base * np.cos(phi)], [R_base * np.sin(phi)], 'o')

print xS_mod0
print yS_mod0

# ax.plot([outx_upper[-1],start_x_relaxed,outx_upper[start_i_relaxed]],[outy_upper[-1],start_y_relaxed,R_[start_i_relaxed]],'x')

# ax.plot(xs,ys,'b')
# ax2.plot(xS,yS,'--')
##ax.legend(['Form','dx/du'])
##plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('dS and ds calculation')
plt.show()

''' Calculate the Young's Modulus '''

E = []
sigma_s = []
sigma_theta = []

for i in range(0, len(kappa_m)):
    sigma_s.append(P / (2.0 * h * kappa_theta[i]))
    temp = (2.0 - kappa_m[i] / kappa_theta[i]) * P / (2.0 * h * kappa_theta[i])
    # if (kappa_m[i]/kappa_theta[i]>1.0):
    # temp = P/(2.0*h*kappa_theta[i])
    sigma_theta.append(temp)
    # temp2 = 10.0
    # if ( ynew[i] > 2.0*tipgrowth_radius):
    # temp2 = (sigma_theta[-1] - nu*sigma_s[-1])/(outy_upper[i]/R_[ds_indizes[i]] - 1.0)
    # temp2 = (sigma_theta[-1] - nu*sigma_s[-1])/(outy_upper[i]/R_fct(S[i]) - 1.0)
    # temp2 = (sigma_theta[-1] - nu*sigma_s[-1])/(outy_upper[ds_indizes[i]]/R_fct(S[i]) - 1.0)

    # temp2 = (sigma_theta[-1] - nu*sigma_s[-1])/(outy_upper[i]/R_[i] - 1.0)
    if (ys[i] / yS[i] > 1.1):
        temp2 = (sigma_theta[-1] - nu * sigma_s[-1]) / (ys[i] / yS[i] - 1.0)
    else:
        temp2 = 5.0

        # if ( S[i]> L - R_base):
        # dist = np.sqrt(outy_upper[i]**2 + (outx_upper[i]- L + r_extended)**2)
        # temp2 = (sigma_theta[transition_index] - nu*sigma_s[transition_index])/(dist/R_base - 1.0)
        # temp2 = E[-1]


        # if ( outx_upper[i]> 3.0):
        # temp2 = (sigma_theta[-1] - nu*sigma_s[-1])/(outy_upper[i]/R_[ds_indizes[i]] - 1.0)

    E.append(temp2)

E_smooth = []

E_smooth.append(E[0])
E_smooth.append((E[0] + E[1]) / 2.0)

for i in range(0, len(E) - 4):
    E_smooth.append((E[i] + E[i + 1] + E[i + 2]) / 3.0)

E_smooth.append(E[-3])
E_smooth.append(E[-3])

fig = plt.figure(1, figsize=(max(x), max(y)), dpi=100)

ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = ax.twinx()

print E_smooth
print max(E_smooth)
print min(E_smooth)

ax.plot(x_points, E_smooth, 'b')
ax2.plot(outx_upper, outy_upper, '--')
# ax.legend(['Form','dx/du'])
# plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Youngs Modulus')
plt.show()
