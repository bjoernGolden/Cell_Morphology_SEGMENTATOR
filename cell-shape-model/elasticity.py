import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Elasticity:
  # Interpolate Splines

  youngs_modulus = []
  params = []
  
  def __init__(self,params,strain):
    ''' Calculate the elasticity '''
    
    self.params = params
    self.youngs_modulus = []
    
    for i in range(0,len(strain.strain_m)):
      
      sigma_m = params.P/(2.0*params.h*strain.kappa_theta[i][1])
      sigma_theta = sigma_m*(2.0 - strain.kappa_m[i][1]/strain.kappa_theta[i][1])
      
      E = (sigma_theta - params.nu*sigma_m)/strain.strain_theta[i]
      self.youngs_modulus.append([strain.naturalShape[i][0],E])
    
  
  def plot_elasticity(self,spline,strain):
    ''' Plot the youngs modulus '''
   
    out = np.array(self.youngs_modulus)
    x = out[:, 0]
    y = out[:, 1]
    
    plt.figure()
    plt.plot(x, y, 'x')
    plt.title("Youngs Modulus")
    plt.show()  
    
    
    naturalShape = strain.naturalShape
    
    wholeShape = spline.spline
    wholeShape_UpperHalf = []
    
    for i in range(1, len(wholeShape[0]) - 1):
      if (wholeShape[1][i] > 0.0):
	wholeShape_UpperHalf.append([wholeShape[0][i],wholeShape[1][i]])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    theta_grid = np.linspace(0,2*np.pi,100)
    out = np.array(naturalShape)
    shape_x = out[:, 0]
    shape_y = out[:, 1]
    
    out = np.array(wholeShape_UpperHalf)
    wholeShape_x = out[:, 0]
    wholeShape_y = out[:, 1]
    
    r_points,theta_points = np.meshgrid(shape_y,theta_grid)
    r_wholeShape,theta_wholeShape = np.meshgrid(wholeShape_y,theta_grid)
    
    temp = [self.youngs_modulus[-1][0],self.youngs_modulus[-1][1]]
    self.youngs_modulus.append(temp)
    out = np.array(self.youngs_modulus)
    

    E_x = out[:, 0]
    E_y = out[:, 1]
    data, theta_points = np.meshgrid(E_y,theta_grid)
    
    print "Length E_y"
    print len(E_y)

    x_points,y_points = r_points*np.cos(theta_points),r_points*np.sin(theta_points)
    x_wholeShape,y_wholeShape = r_wholeShape*np.cos(theta_wholeShape),r_wholeShape*np.sin(theta_wholeShape)
    
    #ax.plot_surface(x_wholeShape, y_wholeShape, wholeShape_x, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
    #plt.show()

    #ax.plot_surface(x_points, y_points, shape_x, rstride=1, cstride=1, cmap=cm.YlGnBu_r)

    #plt.show()

    E_range = max(E_y) - min(E_y)
    #print max(E_y)
    #print min(E_y)
    #N = data/E_range
    #N = data/E_range
    
    max_base = 0.0
    max_trans = 0.0
    max_shaft = 0.0
    max_tip = 0.0
    min_base = 100.0
    min_trans = 100.0
    min_shaft = 100.0
    min_tip = 100.0
    
    for i in range(0,len(self.youngs_modulus)):
      
      if(self.youngs_modulus[i][0]< self.params.r_base): 
	if ( max_base < self.youngs_modulus[i][1] ):
	  max_base = self.youngs_modulus[i][1]
	if ( min_base > self.youngs_modulus[i][1] ):
	  min_base = self.youngs_modulus[i][1]
	  
      if((self.params.r_base < self.youngs_modulus[i][0]) & (self.youngs_modulus[i][0] < self.params.L - self.params.r_shaft - self.params.r_tip)): 
	if ( max_trans < self.youngs_modulus[i][1] ):
	  max_trans = self.youngs_modulus[i][1]
	if ( min_trans > self.youngs_modulus[i][1] ):
	  min_trans = self.youngs_modulus[i][1]
	  
      if((self.params.L - self.params.r_shaft - self.params.r_tip < self.youngs_modulus[i][0]) & (self.youngs_modulus[i][0] < self.params.L - self.params.r_tip)): 
	if ( max_shaft < self.youngs_modulus[i][1] ):
	  max_shaft = self.youngs_modulus[i][1]
	if ( min_shaft > self.youngs_modulus[i][1] ):
	  min_shaft = self.youngs_modulus[i][1]
	  
      if( self.params.L - self.params.r_tip < self.youngs_modulus[i][0] ): 
	if ( max_tip < self.youngs_modulus[i][1] ):
	  max_tip = self.youngs_modulus[i][1]
	if ( min_shaft > self.youngs_modulus[i][1] ):
	  min_tip = self.youngs_modulus[i][1]
	  
    
    print "Youngs Modulus maxima and minima in different regions"
    print "Region\tMax\tMin"
    print "Global\t" + str(max(E_y)) + "\t" + str(min(E_y))
    print "Base\t" + str(max_base) + "\t" + str(min_base)
    print "Neck\t" + str(max_trans) + "\t" + str(min_trans)
    print "Base\t" + str(max_shaft) + "\t" + str(min_shaft)
    print "Tip\t" + str(max_tip) + "\t" + str(min_tip)
    
    
    N = data/5.0
    #print data

    #print N
    #print cm.jet(N) 

    #ax.plot_surface(x_wholeShape, y_wholeShape, wholeShape_x, rstride=1, cstride=1, cmap=cm.Reds)
    ax.plot_surface(x_points, y_points, shape_x, rstride=1, cstride=1, facecolors = cm.Greens_r(N),linewidth=0, antialiased=False, shade=False)
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
	
    

    m = cm.ScalarMappable(cmap=cm.Greens_r)
    m.set_array(N)
    m.set_clim(0.0,5.0)
    plt.colorbar(m)   
    plt.show()
    
    plt.savefig( "Youngs_Modulus.png")
    plt.savefig( "Youngs_Modulus.eps", format="eps")
    
    