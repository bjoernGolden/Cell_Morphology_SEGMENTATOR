import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate

tipgrowth_radius = 0.2
radius_time0 = 3.0

#P = 2.0
#h = 0.01
P = 1.0
h = 0.1
nu = 0.5
r_old = 2.0
r_extended = 3.0
L = 8.0


points = [[0, 0.0], [0.3, 0.7], [1.0, 1.0],  [2.0, 1.3],  [3.0, 1.7], #[L - r_extended*(1.0 - np.cos(4.0*np.pi/3.0)),r_extended*np.sin(4.0*np.pi/3.0)],
	  [L-r_extended,r_extended],[L - r_extended*(1.0 - np.cos(np.pi/3.0)),r_extended*np.sin(np.pi/3.0)],[L - r_extended*(1.0 - np.cos(np.pi/4.0)),r_extended*np.sin(np.pi/4.0)],[L-r_extended*(1.0 - np.cos(np.pi/8.0)),r_extended*np.sin(np.pi/8.0)],[L,0.0]]
temp = len(points) -1
for i in range(1,temp):
  x = points[temp-i][0]
  y = -points[temp-i][1]
  points.append([x,y])

points.append([0, 0.0])

def R_fct(x):
  R = tipgrowth_radius
  
  if( (x <=  L - r_old - r_extended) ):
    R = tipgrowth_radius
  elif (x <=  L + r_old - r_extended ):
    temp = np.sqrt(r_old**2 - (x - L + r_extended)**2)
    R = temp

  return R
    


#t = np.arange(0,1.1,.1)
#x_old = r_old*np.sin(2*np.pi*t)
#for i in range(0,len(x_old)):
  #x_old[i] += L - r_extended
#y_old = r_old*np.cos(2*np.pi*t)
#tck_old,u_old = interpolate.splprep([x_old,y_old],s=0)

#points = [[0, 0.0], [0.001, 0.1],[0.3, 0.7], [1.0, 1.0],  [2.0, 1.3],  [3.0, 1.5],[5.0,3.0],[7.0,2.0],[7.5,1.0],[8.0,0.0]]
#temp = len(points) -1
#for i in range(1,temp):
  #x = points[temp-i][0]
  #y = -points[temp-i][1]
  #points.append([x,y])

#points.append([0, 0.0])
  
points = np.array(points)
x = points[:,0]
y = points[:,1]
tck,u = interpolate.splprep([x,y],k=5,s=0.1)
unew = np.arange(0,1.01,0.01)
out = interpolate.splev(unew,tck)

x_old = L - r_extended + r_old*np.sin(2*np.pi*unew)
#for i in range(0,len(x_old)):
  #x_old[i] += L - r_extended
y_old = r_old*np.cos(2*np.pi*unew)

plt.figure()
plt.plot(x,y,'x',out[0],out[1],x_old,y_old,'b')
#plt.legend(['Linear','Cubic Spline', 'True'])


#plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Spline of parametrically-defined curve')
plt.show()

''' Calcualte Derivatives '''

out_dev = interpolate.splev(unew,tck,der=1)
out_dev2 = interpolate.splev(unew,tck,der=2)

''' cut lower half '''

outx_upper = []
outy_upper = []
devx_upper = []
devy_upper = []
dev2x_upper = []
dev2y_upper = []
R_ = []

epsilon = 0.01

for i in range(0,len(out[0])-1):
  #if( (out[1][i]> tipgrowth_radius + epsilon) & (out[0][i] < L - 1.0) ):
  if( out[1][i]>= 0.0):
    outx_upper.append(out[0][i])
    outy_upper.append(out[1][i])
    devx_upper.append(out_dev[0][i])
    devy_upper.append(out_dev[1][i])
    dev2x_upper.append(out_dev2[0][i])
    dev2y_upper.append(out_dev2[1][i])
    R_.append(R_fct(out[0][i]))
    


fig = plt.figure(1,figsize=(max(x),max(y)),dpi=100)

ax = fig.add_axes([0.1,0.1,0.8,0.8])
#ax2 = ax.twinx()



ax.plot(outx_upper,outy_upper,'b',outx_upper,R_)
#ax2.plot(outx_upper,devy_upper,'--')
#ax.legend(['Form','dx/du'])
#plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Spline of parametrically-defined curve')
plt.show()

''' Calculate meridional curvature '''

kappa_m = []

for i in range(0,len(outx_upper)):
  sign = 1.0
  if ( devx_upper[i] < 0.0 ):	
    sign = -1.0
  nominator = - sign*(dev2y_upper[i]*devx_upper[i] - dev2x_upper[i]*devy_upper[i])
  denominator = np.sqrt(devx_upper[i]**2 + devy_upper[i]**2)**3
  temp = nominator/denominator
  kappa_m.append(temp)
  
  
fig = plt.figure(1,figsize=(max(x),max(y)),dpi=100)

ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax2 = ax.twinx()



ax.plot(outx_upper,kappa_m,'b')
ax2.plot(outx_upper,outy_upper,'--')
#ax.legend(['Form','dx/du'])
#plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Meridional Curvature')
plt.show()


''' Calculate circumferential curvature '''

kappa_theta = []

for i in range(0,len(outx_upper)):
  nominator = devx_upper[i]
  denominator = np.absolute(outy_upper[i])*np.sqrt(devx_upper[i]**2 + devy_upper[i]**2)
  temp = nominator/denominator
  kappa_theta.append(temp)
  
quotient = []
for i in range(0,len(outx_upper)):
  quotient.append(kappa_m[i]/kappa_theta[i])
  

  
fig = plt.figure(1,figsize=(max(x),max(y)),dpi=100)

ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax2 = ax.twinx()



ax.plot(outx_upper,kappa_theta,'b')
ax2.plot(outx_upper,outy_upper,'--')
#ax.legend(['Form','dx/du'])
#plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('Circumferential curvature')
plt.show()


''' Calculate Stresses '''

sigma_s = []
sigma_theta = []

for i in range(0,len(outx_upper)):
  
  sigma_s.append(P/(2.0*h*kappa_theta[i]))
  temp = (2.0 - kappa_m[i]/kappa_theta[i])*P/(2.0*h*kappa_theta[i])  
  if (kappa_m[i]/kappa_theta[i]>1.0):
    temp = P/(2.0*h*kappa_theta[i])
  sigma_theta.append(temp)
  
sigma_s[-1] = max(max(sigma_theta),max(sigma_s))
sigma_theta[-1] = max(max(sigma_theta),max(sigma_s))

sigma_s[-2] = 0.0
sigma_theta[-2] = 0.0

sigma_s[0] = sigma_s[1]
sigma_theta[0] = sigma_theta[1]


fig = plt.figure(1,figsize=(max(x),max(y)),dpi=100)

ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax2 = ax.twinx()


ax.plot(outx_upper,sigma_s,'b')
ax2.plot(outx_upper,sigma_theta,'--')
#ax.legend(['Form','dx/du'])
#plt.axis([-1.05,1.05,-1.05,1.05])
plt.title('sigma_s and sigma_theta')
plt.show()


sigma_s.insert(0, sigma_s[0])
sigma_theta.insert(0, sigma_theta[0])
outy_upper.insert(0,0.0)
outx_upper.insert(0,0.0)

sigma_max = max(max(sigma_theta),max(sigma_s))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

theta_grid = np.linspace(0,2*np.pi,100)
r_points,theta_points = np.meshgrid(outy_upper,theta_grid)
data,theta_points = np.meshgrid(sigma_theta,theta_grid)

x_points,y_points = r_points*np.cos(theta_points),r_points*np.sin(theta_points)

#ax.plot_surface(x_points, y_points, outx_upper, rstride=1, cstride=1, cmap=cm.YlGnBu_r)

N = data/sigma_max

#print N
#print cm.jet(N) 

ax.plot_surface(x_points, y_points, outx_upper, rstride=1, cstride=1, facecolors = cm.coolwarm(N),linewidth=0, antialiased=False, shade=False)
#ax.set_zlim3d(0, cell_radius + shmoo_length)
#ax.set_xlabel(r'$x$')
#ax.set_ylabel(r'$y$')
#ax.set_zlabel(r'$z$')
ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines()+a.get_ticklabels():
        t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)

m = cm.ScalarMappable(cmap=cm.coolwarm)
m.set_array(N)
plt.colorbar(m)   
    
plt.show()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

theta_grid = np.linspace(0,2*np.pi,100)
r_points,theta_points = np.meshgrid(outy_upper,theta_grid)
data,theta_points = np.meshgrid(sigma_s,theta_grid)

x_points,y_points = r_points*np.cos(theta_points),r_points*np.sin(theta_points)

#ax.plot_surface(x_points, y_points, outx_upper, rstride=1, cstride=1, cmap=cm.YlGnBu_r)


#N = data/data.max()
N = data/sigma_max

print N
print cm.jet(N) 

ax.plot_surface(x_points, y_points, outx_upper, rstride=1, cstride=1, facecolors = cm.coolwarm(N),linewidth=0, antialiased=False, shade=False)
#ax.set_zlim3d(0, cell_radius + shmoo_length)
#ax.set_xlabel(r'$x$')
#ax.set_ylabel(r'$y$')
#ax.set_zlabel(r'$z$')

ax.grid(False)
for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
    for t in a.get_ticklines()+a.get_ticklabels():
        t.set_visible(False)
    a.line.set_visible(False)
    a.pane.set_visible(False)

m = cm.ScalarMappable(cmap=cm.coolwarm)
m.set_array(N)
plt.colorbar(m)   
    


plt.show()




