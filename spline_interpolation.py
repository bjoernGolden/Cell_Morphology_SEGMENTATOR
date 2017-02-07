import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

class Spline_Interpolation:
  ''' Interpolate natural cell shape with spline 
      and obtain curvatures '''
      
  pointsNaturalShape = []
  pointsWeight = []
  u = []
  tck = []
  spline = []
  Dspline_Du = []
  D2spline_D2u = []
  kappa_m = []
  kappa_theta = []
  
  
  
  def __init__(self,points,weightAr,path):
    
    self.path = path
    for pt in points:
      self.pointsNaturalShape.append(pt)
    for pwt in weightAr:
      self.pointsWeight.append(pwt)
      
    temp = len(points) - 1  
    
    for i in range(0, len(self.pointsNaturalShape)):
      xta = -self.pointsNaturalShape[temp - i][0]
      yta = self.pointsNaturalShape[temp - i][1]
      self.pointsNaturalShape.append([xta, yta])
      w = self.pointsWeight[temp-i]
      self.pointsWeight.append(w)
    print "len pointsNatShape: ", len(self.pointsNaturalShape)
    print "len PointsWeight: ", len(self.pointsWeight)

    #self.pointsNaturalShape.append(self.pointsNaturalShape[0])
    #self.pointsNaturalShape.insert(0, [0, 0.0])	
    #print "Points1: ",self.pointsNaturalShape
    lengthNatShape = len(self.pointsNaturalShape)
    lengthpointsWeight = len(self.pointsWeight)
    partLength = int(round(lengthNatShape/4))
    partLengthWeight = int (round(lengthpointsWeight/4))
    hArray = self.pointsNaturalShape[-partLength:]
    hArrayWeight = self.pointsWeight[-partLengthWeight:]

    del self.pointsNaturalShape[-partLength:]
    del self.pointsWeight[-partLengthWeight:]

    #print "Points2: " ,self.pointsNaturalShape
    
    self.pointsNaturalShape = hArray + self.pointsNaturalShape
    self.pointsWeight = hArrayWeight + self.pointsWeight

    #print "Points3: ",self.pointsNaturalShape
    x = []
    y = []
    self.pointsNaturalShape = np.array(self.pointsNaturalShape)
    self.pointsWeight = np.array(self.pointsWeight)

    x = self.pointsNaturalShape[:, 0]
    y = self.pointsNaturalShape[:, 1]
    #w = ones(len(x))
    
    #for i in range(0,len(w)):
    #   w[i] = 
    print self.pointsWeight
    print np.ones(lengthpointsWeight)
    print "len x0: ", len(x)
    print "len weight: ",len(self.pointsWeight)
    lx = len(x)


    #self.tck, self.u = interpolate.splprep([x, y], k=5, s=0.1)
    self.tck, self.u = interpolate.splprep([x, y],w = self.pointsWeight, k=3, s=50)# 70 s=65 #100
    # unew = np.arange(0,1.01,0.01)
    print "u: ", self.u
    print "tck: ", self.tck
    self.u = np.arange(0, 1.01, 0.005)
    self.spline = interpolate.splev(self.u, self.tck)
    print "spline: ", self.spline
      
    #''' Calcualte Derivatives '''

    self.Dspline_Du = interpolate.splev(self.u, self.tck, der=1)
    self.D2spline_D2u = interpolate.splev(self.u, self.tck, der=2)
    
     
     
  def show_naturalshape(self):
    ''' Plot the spline ''' 
        
    x = self.pointsNaturalShape[:, 0]
    y = self.pointsNaturalShape[:, 1]
    plt.figure()
    plt.plot(y,x,'ro')
    plt.show()

    plt.figure()
    plt.plot(y,x, 'x',  self.spline[1], self.spline[0],'b')
    plt.title('Spline of parametrically-defined curve')
    plt.savefig(self.path + ".png")
    plt.show()

  def show_naturalshapeMicro(self): 
    x = self.pointsNaturalShape[:, 0]*0.13
    y = self.pointsNaturalShape[:, 1]*0.13
    plt.figure()
    plt.plot(y,x,'ro')
    plt.show()
    spline1 = self.spline[1]*0.13
    spline0 = self.spline[0]*0.13
    plt.figure()
    plt.plot(y,x, 'x',  spline1, spline0,'b')
    plt.title('Spline of parametrically-defined curve')
    plt.savefig(self.path + ".png")
    plt.show()
    xneg = []
    yneg = []
    for i in range(np.size(x)):
      if x[i]<0: 
        xneg.append(x[i])
        yneg.append(y[i])
    spline0neg = []
    spline1neg = []
    temp2 = np.size(xneg)-1
    for i in range(np.size(x)):
      xpos = -xneg[temp2-i]
      ypos = yneg[temp2-i]
      xneg.append(xpos)
      yneg.append(ypos)

    for i in range(np.size(spline0)):
      if spline0[i] <= 0.1 : 
        spline0neg.append(spline0[i])
        spline1neg.append(spline1[i])

    temp3 = np.size(spline0neg)-1
    for i in range(np.size(spline0neg)):
        spline0pos = -spline0neg[temp3-i]
        spline1pos = spline1neg[temp3-i]
        spline0neg.append(spline0pos)
        spline1neg.append(spline1pos)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(yneg,xneg,'x',spline1neg,spline0neg,'b')
    ax.set_aspect('equal')
    #ax.axhline(y=0, color='k')
    #ax.axvline(x=0, color='k')
    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_coords(0.87, 0.45)
    ax.yaxis.set_label_coords(0.28, 0.93)
    plt.axis((-5,7 ,-5,5))

    plt.ylabel('width [$\mu$m]', fontsize = 16).set_rotation(0)
    plt.xlabel('length [$\mu$m]', fontsize = 16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
      label.set_fontsize(15)

    #plt.plot(yneg,xneg,'x',spline1neg,spline0neg,'b')
    plt.show()


  def calculate_meridionalCurvature(self,a,b):
    ''' Calculates the meridional curvature in the interval (a,b) '''
    
    self.kappa_m = []
    
    for i in range(1, len(self.spline[0]) - 1):
      if ((a < self.spline[0][i]) & (self.spline[0][i] < b) & (self.spline[1][i] > 0.0)):
	sign = 1.0
	if (self.Dspline_Du[0][i] < 0.0):
	  sign = -1.0
	nominator = - sign * (self.D2spline_D2u[1][i] * self.Dspline_Du[0][i] - self.D2spline_D2u[0][i] * self.Dspline_Du[1][i])
	denominator = np.sqrt(self.Dspline_Du[0][i] ** 2 + self.Dspline_Du[1][i] ** 2) ** 3
	temp = nominator / denominator
	self.kappa_m.append([self.spline[0][i],temp])
	#print temp
	
    out = np.array(self.kappa_m)
	
    x = out[:, 0]
    y = out[:, 1]
    
    plt.figure()
    plt.plot(x, y, 'x', self.spline[0], self.spline[1], 'b')
    plt.title('Meridional curvature along upper half of the shape')
    plt.show()
	  
  def calculate_meridionalCurvature(self,a,b,show=False):
    ''' Calculates the meridional curvature in the interval (a,b) '''
    
    self.kappa_m = []
    
    for i in range(1, len(self.spline[0]) - 1):
      if ((a < self.spline[0][i]) & (self.spline[0][i] < b) & (self.spline[1][i] > 0.0)):
	sign = 1.0
	if (self.Dspline_Du[0][i] < 0.0):
	  sign = -1.0
	nominator = - sign * (self.D2spline_D2u[1][i] * self.Dspline_Du[0][i] - self.D2spline_D2u[0][i] * self.Dspline_Du[1][i])
	denominator = np.sqrt(self.Dspline_Du[0][i] ** 2 + self.Dspline_Du[1][i] ** 2) ** 3
	temp = nominator / denominator
	self.kappa_m.append([self.spline[0][i],temp])
	#print temp
	

    
    if (show):
      out = np.array(self.kappa_m)
      x = out[:, 0]
      y = out[:, 1]
      
      plt.figure()
      plt.plot(x, y, 'x', self.spline[0], self.spline[1], 'b')
      plt.title('Meridional curvature along upper half of the shape')
      plt.show()
    
  def calculate_circumferentialCurvature(self,a,b,show=False):
    ''' Calculates the meridional curvature in the interval (a,b) '''
    
    self.kappa_theta = []
    
    for i in range(1, len(self.spline[0]) - 1):
      if ((a < self.spline[0][i]) & (self.spline[0][i] < b) & (self.spline[1][i] > 0.0)):
	nominator = self.Dspline_Du[0][i]
	denominator = np.absolute(self.spline[1][i]) * np.sqrt(self.Dspline_Du[0][i] ** 2 + self.Dspline_Du[1][i] ** 2)
	temp = nominator / denominator
	self.kappa_theta.append([self.spline[0][i],temp])
	#print temp
	
    out = np.array(self.kappa_theta)
	
    x = out[:, 0]
    y = out[:, 1]
    
    if (show):
      plt.figure()
      plt.plot(x, y, 'x', self.spline[0], self.spline[1], 'b')
      plt.title('Circumferential curvature along upper half of the shape')
      plt.show()