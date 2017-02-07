###Marie Hemmen, 05.09.16###

import sys
from shmooclass import shmoo
from spline_interpolation import Spline_Interpolation
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy
from scipy import misc
from scipy import ndimage
import math
import pdb
import Image
from random import randint
import re
from scipy import pi,sin,cos
from numpy import linspace
from numpy import (array, dot, arccos)
from numpy.linalg import norm
import pylab as pl
import os, sys



np.set_printoptions(threshold=np.nan)


def makeEllipse1(x0,y0,a,b,an):

	points=1000 #Number of points which needs to construct the elipse
	cos_a=cos(an*pi/180.)
	sin_a=sin(an*pi/180.)
	the=linspace(0,2*pi,points)
	X=a*cos(the)*cos_a-sin_a*b*sin(the)+x0
	Y=a*cos(the)*sin_a+cos_a*b*sin(the)+y0
	x_values=np.array([X])
	pos_y_values=np.array([Y])

	array_ellipse = np.append(x_values,pos_y_values, axis = 0)
	return array_ellipse

def makeArray(array,mx,my): 
	modelarray=np.zeros((my,mx)) 
	for m in range(array[0].size):
		x=array[0][m]-1
		y=array[1][m]-1
		if x<0: 
			x = 0
		if y<0: 
			y = 0
		modelarray[y][x]=1
	return modelarray

def angle(pt1, pt2):
	x1, y1 = pt1
	x2, y2 = pt2
	inner_product = x1*x2 + y1*y2
	len1 = math.hypot(x1, y1)
	len2 = math.hypot(x2, y2)
	if len1 >=1 and len2 >= 1:
		return math.acos(inner_product/(len1*len2))
	else:
		return 0

def calculate(pt, pt2):
	ang = angle(pt,pt2)*180/math.pi
	#print "ang: ",ang
	return ang

def rotate(image,rot):
	#print "rotateangle: ", rot
	rad = np.radians(rot)
	a = np.cos(rad)
	b = np.sin(rad)
	R = np.mat([[a, -b], [b,a]])
	#print "R: ", R
	Y = np.array(R*image) # rotation and scaling
	return Y

def MakeImage(array,titel): 
	plt.imshow(array,cmap = "YlGnBu") #YlGnBu
	plt.colorbar()
	plt.title(titel)

def MakeImagetr(array,title,x0t,y0t):
	print "x0av: ", x0t, "y0av: ", y0t
	h, w = array.shape
	ax = plt.gca()

	plt.imshow(array,cmap = "YlGnBu",
			  extent=[-x0t, w-x0t, h-y0t, -y0t]) #YlGnBu
	#plt.colorbar()
	
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('#Number of Cells')
	ax.xaxis.set_label_coords(0.89, 0.45)
	ax.yaxis.set_label_coords(0.33, 0.93)

	plt.xlabel('Width [Pixel]', fontsize = 13)
	plt.ylabel('Length [Pixel]', fontsize = 13).set_rotation(0)



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
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_fontsize(12)

	plt.title(title, fontsize = 16)


def MakeImagetrMicro(array,title,x0t2,y0t2):
	print "x0av: ", x0t2, "y0av: ", y0t2
	h, w = array.shape
	ax = plt.gca()

	plt.imshow(array,cmap = "YlGnBu",
			  extent=[-x0t2*0.13, (w-x0t2)*0.13, (h-y0t2)*0.13, -y0t2*0.13]) #YlGnBu
	#plt.colorbar()
	
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('#Number of Cells')
	ax.xaxis.set_label_coords(0.89, 0.45)
	ax.yaxis.set_label_coords(0.3, 0.93)

	plt.xlabel('Width [$\mu$m]', fontsize = 13)
	plt.ylabel('Length [$\mu$m]', fontsize = 13).set_rotation(0)



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
	plt.axis((-5,7 ,-5,5))

	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_fontsize(12)

	plt.title(title, fontsize = 16)

def MakeNormedImage(array,title):
	plt.imshow(array,cmap = "YlGnBu",vmin=0, vmax = 0.4)
	plt.colorbar()

def MakeNormalizedImage(array,title,x0t3,y0t3):
	h, w = array.shape
	ax = plt.gca()
	plt.imshow(array,cmap = "YlGnBu",
				extent=[-x0t3*0.13, (w-x0t3)*0.13, (h-y0t3)*0.13, -y0t3*0.13])
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('normalized frequency of cell outlines', fontsize = 16)
	ax.xaxis.set_label_coords(0.86, 0.43)
	ax.yaxis.set_label_coords(0.28, 0.93)

	plt.ylabel('width [$\mu$m]',fontsize = 30).set_rotation(0)
	plt.xlabel('length [$\mu$m]',fontsize = 30)
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
	plt.axis((-5,7 ,-5,5))

	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_fontsize(25)

	plt.title(title)
def MakeImage2(array1,array2, array3): 
	plt.imshow(array1,cmap = "gray", alpha = 0.5)
	plt.imshow(array2,cmap = "gray", alpha = 0.5)
	plt.imshow(array3,cmap = "gray", alpha = 0.5)

def ShowImage(path): 
	plt.draw()
	plt.savefig(path + ".png")
	plt.show()


parameterfileIn = sys.argv[1] #/home/marie/Master/EllipseparametersNew/EKY360
imagefileIn = sys.argv[2] #/home/marie/Master/Outlinecoordinates/Positives_new
ImageLocOut = sys.argv[3] #/home/marie/Master/Average_Images_New/EKY360Sorted
Matlabfile = sys.argv[4] #/home/marie/Master/Average_Images_New/MatlabFiles/EKY360
valuesfile = sys.argv[5] #/home/marie/Master/Values/EKY360/160
Strain = sys.argv[6] #EKY360
MatlabfileNorm = sys.argv[7] #/home/marie/Master/Average_Images_New/MatlabFilesNorm/EKY360


wf = open(valuesfile,'w')

files = os.listdir(parameterfileIn)
alllengths = []
#max_x = 110
#max_y = 90

for fi in files: 
	pfile = open(parameterfileIn+"/"+fi, 'r')
	pfile.readline()
	#print pfile
	for fline in pfile:
		fline = fline.split("\t")
		areaf = float(fline[5])
		if areaf<500: 
			continue
		length = float(fline[8])
		alllengths.append(length)
	#pfile.close()


minlength = np.amin(alllengths)
maxlength = np.amax(alllengths)
print "minlength: ", minlength
print "maxlength: ", maxlength

max_x = 120 #100
max_y = 100 #80

minlength = 20.0
dif = 4.42


l1 = minlength + dif
l2 = l1+dif
l3 = l2+dif
l4 = l3+dif
l5 = l4+dif
l6 = l5+dif
l7 = l6+dif
l8 = l7+dif
l9 = l8+dif
l10 = l9+dif
l11 = l10+dif
l12 = l11+dif

minlengthMicro = minlength * 0.13
l2Micro = l2*0.13
l3Micro = l3*0.13
l4Micro = l4*0.13
l5Micro = l5*0.13
l6Micro = l6*0.13
l7Micro = l7*0.13
l8Micro = l8*0.13
l9Micro = l9*0.13
l10Micro = l10*0.13
l11Micro = l11*0.13
l12Micro = l12*0.13
l1Micro = l1*0.13


print minlength, maxlength, l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12



a_averagelistl1 = []
b_averagelistl1 = []
a2_averagelistl1 = []
b2_averagelistl1 = []
area_averagelistl1 = []
perimeter_averagelistl1 = []
length_averagelistl1 = []
x0_averagelistl1 = []
y0_averagelistl1 = []
x02_averagelistl1 = []
y02_averagelistl1 = []
aMikro_averagelistl1 = []
bMikro_averagelistl1 = []
a2Mikro_averagelistl1 = []
b2Mikro_averagelistl1 = []
areaMikro_averagelistl1 = []
perimeterMikro_averagelistl1 = []
lengthMikro_averagelistl1 = []
x0Mikro_averagelistl1 = []
y0Mikro_averagelistl1 = []

a_averagelistl2 = []
b_averagelistl2 = []
a2_averagelistl2 = []
b2_averagelistl2 = []
area_averagelistl2 = []
perimeter_averagelistl2 = []
length_averagelistl2 = []
x0_averagelistl2 = []
y0_averagelistl2 = []
x02_averagelistl2 = []
y02_averagelistl2 = []
aMikro_averagelistl2 = []
bMikro_averagelistl2 = []
a2Mikro_averagelistl2 = []
b2Mikro_averagelistl2 = []
areaMikro_averagelistl2 = []
perimeterMikro_averagelistl2 = []
lengthMikro_averagelistl2 = []
x0Mikro_averagelistl2 = []
y0Mikro_averagelistl2 = []

a_averagelistl3 = []
b_averagelistl3 = []
a2_averagelistl3 = []
b2_averagelistl3 = []
area_averagelistl3 = []
perimeter_averagelistl3 = []
length_averagelistl3 = []
x0_averagelistl3 = []
y0_averagelistl3 = []
x02_averagelistl3 = []
y02_averagelistl3 = []
aMikro_averagelistl3 = []
bMikro_averagelistl3 = []
a2Mikro_averagelistl3 = []
b2Mikro_averagelistl3 = []
areaMikro_averagelistl3 = []
perimeterMikro_averagelistl3 = []
lengthMikro_averagelistl3 = []
x0Mikro_averagelistl3 = []
y0Mikro_averagelistl3 = []

a_averagelistl4 = []
b_averagelistl4 = []
a2_averagelistl4 = []
b2_averagelistl4 = []
area_averagelistl4 = []
perimeter_averagelistl4 = []
length_averagelistl4 = []
x0_averagelistl4 = []
y0_averagelistl4 = []
x02_averagelistl4 = []
y02_averagelistl4 = []
aMikro_averagelistl4 = []
bMikro_averagelistl4 = []
a2Mikro_averagelistl4 = []
b2Mikro_averagelistl4 = []
areaMikro_averagelistl4 = []
perimeterMikro_averagelistl4 = []
lengthMikro_averagelistl4 = []
x0Mikro_averagelistl4 = []
y0Mikro_averagelistl4 = []

a_averagelistl5 = []
b_averagelistl5 = []
a2_averagelistl5 = []
b2_averagelistl5 = []
area_averagelistl5 = []
perimeter_averagelistl5 = []
length_averagelistl5 = []
x0_averagelistl5 = []
y0_averagelistl5 = []
x02_averagelistl5 = []
y02_averagelistl5 = []
aMikro_averagelistl5 = []
bMikro_averagelistl5 = []
a2Mikro_averagelistl5 = []
b2Mikro_averagelistl5 = []
areaMikro_averagelistl5 = []
perimeterMikro_averagelistl5 = []
lengthMikro_averagelistl5 = []
x0Mikro_averagelistl5 = []
y0Mikro_averagelistl5 = []

a_averagelistl6 = []
b_averagelistl6 = []
a2_averagelistl6 = []
b2_averagelistl6 = []
area_averagelistl6 = []
perimeter_averagelistl6 = []
length_averagelistl6 = []
x0_averagelistl6 = []
y0_averagelistl6 = []
x02_averagelistl6 = []
y02_averagelistl6 = []
aMikro_averagelistl6 = []
bMikro_averagelistl6 = []
a2Mikro_averagelistl6 = []
b2Mikro_averagelistl6 = []
areaMikro_averagelistl6 = []
perimeterMikro_averagelistl6 = []
lengthMikro_averagelistl6 = []
x0Mikro_averagelistl6 = []
y0Mikro_averagelistl6 = []

a_averagelistl7 = []
b_averagelistl7 = []
a2_averagelistl7 = []
b2_averagelistl7 = []
area_averagelistl7 = []
perimeter_averagelistl7 = []
length_averagelistl7 = []
x0_averagelistl7 = []
y0_averagelistl7 = []
x02_averagelistl7 = []
y02_averagelistl7 = []
aMikro_averagelistl7 = []
bMikro_averagelistl7 = []
a2Mikro_averagelistl7 = []
b2Mikro_averagelistl7 = []
areaMikro_averagelistl7 = []
perimeterMikro_averagelistl7 = []
lengthMikro_averagelistl7 = []
x0Mikro_averagelistl7 = []
y0Mikro_averagelistl7 = []

a_averagelistl8 = []
b_averagelistl8 = []
a2_averagelistl8 = []
b2_averagelistl8 = []
area_averagelistl8 = []
perimeter_averagelistl8 = []
length_averagelistl8 = []
x0_averagelistl8 = []
y0_averagelistl8 = []
x02_averagelistl8 = []
y02_averagelistl8 = []
aMikro_averagelistl8 = []
bMikro_averagelistl8 = []
a2Mikro_averagelistl8 = []
b2Mikro_averagelistl8 = []
areaMikro_averagelistl8 = []
perimeterMikro_averagelistl8 = []
lengthMikro_averagelistl8 = []
x0Mikro_averagelistl8 = []
y0Mikro_averagelistl8 = []

a_averagelistl9 = []
b_averagelistl9 = []
a2_averagelistl9 = []
b2_averagelistl9 = []
area_averagelistl9 = []
perimeter_averagelistl9 = []
length_averagelistl9 = []
x0_averagelistl9 = []
y0_averagelistl9 = []
x02_averagelistl9 = []
y02_averagelistl9 = []
aMikro_averagelistl9 = []
bMikro_averagelistl9 = []
a2Mikro_averagelistl9 = []
b2Mikro_averagelistl9 = []
areaMikro_averagelistl9 = []
perimeterMikro_averagelistl9 = []
lengthMikro_averagelistl9 = []
x0Mikro_averagelistl9 = []
y0Mikro_averagelistl9 = []

a_averagelistl10 = []
b_averagelistl10 = []
a2_averagelistl10 = []
b2_averagelistl10 = []
area_averagelistl10 = []
perimeter_averagelistl10 = []
length_averagelistl10 = []
x0_averagelistl10 = []
y0_averagelistl10 = []
x02_averagelistl10 = []
y02_averagelistl10 = []
aMikro_averagelistl10 = []
bMikro_averagelistl10 = []
a2Mikro_averagelistl10 = []
b2Mikro_averagelistl10 = []
areaMikro_averagelistl10 = []
perimeterMikro_averagelistl10 = []
lengthMikro_averagelistl10 = []
x0Mikro_averagelistl10 = []
y0Mikro_averagelistl10 = []

a_averagelistl11 = []
b_averagelistl11 = []
a2_averagelistl11 = []
b2_averagelistl11 = []
area_averagelistl11 = []
perimeter_averagelistl11 = []
length_averagelistl11 = []
x0_averagelistl11 = []
y0_averagelistl11 = []
x02_averagelistl11 = []
y02_averagelistl11 = []
aMikro_averagelistl11 = []
bMikro_averagelistl11 = []
a2Mikro_averagelistl11 = []
b2Mikro_averagelistl11 = []
areaMikro_averagelistl11 = []
perimeterMikro_averagelistl11 = []
lengthMikro_averagelistl11 = []
x0Mikro_averagelistl11 = []
y0Mikro_averagelistl11 = []

a_averagelistl12 = []
b_averagelistl12 = []
a2_averagelistl12 = []
b2_averagelistl12 = []
area_averagelistl12 = []
perimeter_averagelistl12 = []
length_averagelistl12 = []
x0_averagelistl12 = []
y0_averagelistl12 = []
x02_averagelistl12 = []
y02_averagelistl12 = []
aMikro_averagelistl12 = []
bMikro_averagelistl12 = []
a2Mikro_averagelistl12 = []
b2Mikro_averagelistl12 = []
areaMikro_averagelistl12 = []
perimeterMikro_averagelistl12 = []
lengthMikro_averagelistl12 = []
x0Mikro_averagelistl12 = []
y0Mikro_averagelistl12 = []



allarrayreducedl1 = np.zeros((max_y,max_x))
allarrayreducedl2 = np.zeros((max_y,max_x))
allarrayreducedl3 = np.zeros((max_y,max_x))
allarrayreducedl4 = np.zeros((max_y,max_x))
allarrayreducedl5 = np.zeros((max_y,max_x))
allarrayreducedl6 = np.zeros((max_y,max_x))
allarrayreducedl7 = np.zeros((max_y,max_x))
allarrayreducedl8 = np.zeros((max_y,max_x))
allarrayreducedl9 = np.zeros((max_y,max_x))
allarrayreducedl10 = np.zeros((max_y,max_x))
allarrayreducedl11 = np.zeros((max_y,max_x))
allarrayreducedl12 = np.zeros((max_y,max_x))

allarrayl1 = np.zeros((max_y,max_x)) 
allarrayl2 = np.zeros((max_y,max_x)) 
allarrayl3 = np.zeros((max_y,max_x)) 
allarrayl4 = np.zeros((max_y,max_x)) 
allarrayl5 = np.zeros((max_y,max_x)) 
allarrayl6 = np.zeros((max_y,max_x)) 
allarrayl7 = np.zeros((max_y,max_x)) 
allarrayl8 = np.zeros((max_y,max_x)) 
allarrayl9 = np.zeros((max_y,max_x)) 
allarrayl10 = np.zeros((max_y,max_x)) 
allarrayl11 = np.zeros((max_y,max_x)) 
allarrayl12 = np.zeros((max_y,max_x)) 


allcellsl1 = np.zeros((max_y,max_x))
allcellsPosl1 = np.zeros((max_y,max_x))
allcellsFHSpll1 = np.zeros((max_y,max_x))
allcellsSHSpll1 = np.zeros((max_y,max_x))
allcellsl2 = np.zeros((max_y,max_x))
allcellsPosl2 = np.zeros((max_y,max_x))
allcellsFHSpll2 = np.zeros((max_y,max_x))
allcellsSHSpll2 = np.zeros((max_y,max_x))
allcellsl3 = np.zeros((max_y,max_x))
allcellsPosl3 = np.zeros((max_y,max_x))
allcellsFHSpll3 = np.zeros((max_y,max_x))
allcellsSHSpll3 = np.zeros((max_y,max_x))
allcellsl4 = np.zeros((max_y,max_x))
allcellsPosl4 = np.zeros((max_y,max_x))
allcellsFHSpll4 = np.zeros((max_y,max_x))
allcellsSHSpll4 = np.zeros((max_y,max_x))
allcellsl5 = np.zeros((max_y,max_x))
allcellsPosl5 = np.zeros((max_y,max_x))
allcellsFHSpll5 = np.zeros((max_y,max_x))
allcellsSHSpll5 = np.zeros((max_y,max_x))
allcellsl6 = np.zeros((max_y,max_x))
allcellsPosl6 = np.zeros((max_y,max_x))
allcellsFHSpll6 = np.zeros((max_y,max_x))
allcellsSHSpll6 = np.zeros((max_y,max_x))
allcellsl7 = np.zeros((max_y,max_x))
allcellsPosl7 = np.zeros((max_y,max_x))
allcellsFHSpll7 = np.zeros((max_y,max_x))
allcellsSHSpll7 = np.zeros((max_y,max_x))
allcellsl8 = np.zeros((max_y,max_x))
allcellsPosl8 = np.zeros((max_y,max_x))
allcellsFHSpll8 = np.zeros((max_y,max_x))
allcellsSHSpll8 = np.zeros((max_y,max_x))
allcellsl9 = np.zeros((max_y,max_x))
allcellsPosl9 = np.zeros((max_y,max_x))
allcellsFHSpll9 = np.zeros((max_y,max_x))
allcellsSHSpll9 = np.zeros((max_y,max_x))
allcellsl10 = np.zeros((max_y,max_x))
allcellsPosl10 = np.zeros((max_y,max_x))
allcellsFHSpll10 = np.zeros((max_y,max_x))
allcellsSHSpll10 = np.zeros((max_y,max_x))
allcellsl11 = np.zeros((max_y,max_x))
allcellsPosl11 = np.zeros((max_y,max_x))
allcellsFHSpll11 = np.zeros((max_y,max_x))
allcellsSHSpll11 = np.zeros((max_y,max_x))
allcellsl12 = np.zeros((max_y,max_x))
allcellsPosl12 = np.zeros((max_y,max_x))
allcellsFHSpll12 = np.zeros((max_y,max_x))
allcellsSHSpll12 = np.zeros((max_y,max_x))
allcellsl5Spline = np.zeros((max_y,max_x))

allcellsFHl1 = np.zeros((max_y,max_x))
allcellsSHl1 = np.zeros((max_y,max_x))
allcellsFHl2= np.zeros((max_y,max_x))
allcellsSHl2 = np.zeros((max_y,max_x))
allcellsFHl3 = np.zeros((max_y,max_x))
allcellsSHl3 = np.zeros((max_y,max_x))
allcellsFHl4 = np.zeros((max_y,max_x))
allcellsSHl4= np.zeros((max_y,max_x))
allcellsFHl5 = np.zeros((max_y,max_x))
allcellsSHl5 = np.zeros((max_y,max_x))
allcellsFHl6 = np.zeros((max_y,max_x))
allcellsSHl6 = np.zeros((max_y,max_x))
allcellsFHl7 = np.zeros((max_y,max_x))
allcellsSHl7 = np.zeros((max_y,max_x))
allcellsFHl8 = np.zeros((max_y,max_x))
allcellsSHl8 = np.zeros((max_y,max_x))
allcellsFHl9 = np.zeros((max_y,max_x))
allcellsSHl9 = np.zeros((max_y,max_x))
allcellsFHl10 = np.zeros((max_y,max_x))
allcellsSHl10 = np.zeros((max_y,max_x))
allcellsFHl11 = np.zeros((max_y,max_x))
allcellsSHl11 = np.zeros((max_y,max_x))
allcellsFHl12= np.zeros((max_y,max_x))
allcellsSHl12 = np.zeros((max_y,max_x))

allcellsl1pos = np.zeros((max_y,max_x))
allcellsl1posT= np.zeros((max_y,max_x))
allcellsl1posSpline= np.zeros((max_y,max_x))
allcellsl1neg = np.zeros((max_y,max_x))
allcellsl1negT= np.zeros((max_y,max_x))
allcellsl1negSpline= np.zeros((max_y,max_x))
allcellsl1FH= np.zeros((max_y,max_x))
allcellsl1RH= np.zeros((max_y,max_x))
allcellsl1FHSpl= np.zeros((max_y,max_x))
allcellsl1RHSpl= np.zeros((max_y,max_x))

allcellsl2pos = np.zeros((max_y,max_x))
allcellsl2posT= np.zeros((max_y,max_x))
allcellsl2posSpline= np.zeros((max_y,max_x))
allcellsl2neg = np.zeros((max_y,max_x))
allcellsl2negT= np.zeros((max_y,max_x))
allcellsl2negSpline= np.zeros((max_y,max_x))
allcellsl2FH= np.zeros((max_y,max_x))
allcellsl2RH= np.zeros((max_y,max_x))
allcellsl2FHSpl= np.zeros((max_y,max_x))
allcellsl2RHSpl= np.zeros((max_y,max_x))

allcellsl3pos = np.zeros((max_y,max_x))
allcellsl3posT= np.zeros((max_y,max_x))
allcellsl3posSpline= np.zeros((max_y,max_x))
allcellsl3neg = np.zeros((max_y,max_x))
allcellsl3negT= np.zeros((max_y,max_x))
allcellsl3negSpline= np.zeros((max_y,max_x))
allcellsl3FH= np.zeros((max_y,max_x))
allcellsl3RH= np.zeros((max_y,max_x))
allcellsl3FHSpl= np.zeros((max_y,max_x))
allcellsl3RHSpl= np.zeros((max_y,max_x))

allcellsl4pos = np.zeros((max_y,max_x))
allcellsl4posT= np.zeros((max_y,max_x))
allcellsl4posSpline= np.zeros((max_y,max_x))
allcellsl4neg = np.zeros((max_y,max_x))
allcellsl4negT= np.zeros((max_y,max_x))
allcellsl4negSpline= np.zeros((max_y,max_x))
allcellsl4FH= np.zeros((max_y,max_x))
allcellsl4RH= np.zeros((max_y,max_x))
allcellsl4FHSpl= np.zeros((max_y,max_x))
allcellsl4RHSpl= np.zeros((max_y,max_x))

allcellsl5pos = np.zeros((max_y,max_x))
allcellsl5posT= np.zeros((max_y,max_x))
allcellsl5posSpline= np.zeros((max_y,max_x))
allcellsl5neg = np.zeros((max_y,max_x))
allcellsl5negT= np.zeros((max_y,max_x))
allcellsl5negSpline= np.zeros((max_y,max_x))
BothSplines = np.zeros((max_y,max_x))
BothSplinesRed = np.zeros((max_y,max_x))
allcellsl5FH= np.zeros((max_y,max_x))
allcellsl5RH= np.zeros((max_y,max_x))
allcellsl5FHSpl= np.zeros((max_y,max_x))
allcellsl5RHSpl= np.zeros((max_y,max_x))
BothSplines2 = np.zeros((max_y,max_x))

allcellsl6pos = np.zeros((max_y,max_x))
allcellsl6posT= np.zeros((max_y,max_x))
allcellsl6posSpline= np.zeros((max_y,max_x))
allcellsl6neg = np.zeros((max_y,max_x))
allcellsl6negT= np.zeros((max_y,max_x))
allcellsl6negSpline= np.zeros((max_y,max_x))
allcellsl6FH= np.zeros((max_y,max_x))
allcellsl6RH= np.zeros((max_y,max_x))
allcellsl6FHSpl= np.zeros((max_y,max_x))
allcellsl6RHSpl= np.zeros((max_y,max_x))


allcellsl7pos = np.zeros((max_y,max_x))
allcellsl7posT= np.zeros((max_y,max_x))
allcellsl7posSpline= np.zeros((max_y,max_x))
allcellsl7neg = np.zeros((max_y,max_x))
allcellsl7negT= np.zeros((max_y,max_x))
allcellsl7negSpline= np.zeros((max_y,max_x))
allcellsl7FH= np.zeros((max_y,max_x))
allcellsl7RH= np.zeros((max_y,max_x))
allcellsl7FHSpl= np.zeros((max_y,max_x))
allcellsl7RHSpl= np.zeros((max_y,max_x))
BothSplines7 = np.zeros((max_y,max_x))
BothSplinesRed7 = np.zeros((max_y,max_x))

allcellsl8pos = np.zeros((max_y,max_x))
allcellsl8posT= np.zeros((max_y,max_x))
allcellsl8posSpline= np.zeros((max_y,max_x))
allcellsl8neg = np.zeros((max_y,max_x))
allcellsl8negT= np.zeros((max_y,max_x))
allcellsl8negSpline= np.zeros((max_y,max_x))
allcellsl8FH= np.zeros((max_y,max_x))
allcellsl8RH= np.zeros((max_y,max_x))
allcellsl8FHSpl= np.zeros((max_y,max_x))
allcellsl8RHSpl= np.zeros((max_y,max_x))
BothSplines8 = np.zeros((max_y,max_x))
BothSplinesRed8 = np.zeros((max_y,max_x))

allcellsl9pos = np.zeros((max_y,max_x))
allcellsl9posT= np.zeros((max_y,max_x))
allcellsl9posSpline= np.zeros((max_y,max_x))
allcellsl9neg = np.zeros((max_y,max_x))
allcellsl9negT= np.zeros((max_y,max_x))
allcellsl9negSpline= np.zeros((max_y,max_x))
allcellsl9FH= np.zeros((max_y,max_x))
allcellsl9RH= np.zeros((max_y,max_x))
allcellsl9FHSpl= np.zeros((max_y,max_x))
allcellsl9RHSpl= np.zeros((max_y,max_x))
BothSplines9 = np.zeros((max_y,max_x))
BothSplinesRed9 = np.zeros((max_y,max_x))

allcellsl10pos = np.zeros((max_y,max_x))
allcellsl10posT= np.zeros((max_y,max_x))
allcellsl10posSpline= np.zeros((max_y,max_x))
allcellsl10neg = np.zeros((max_y,max_x))
allcellsl10negT= np.zeros((max_y,max_x))
allcellsl10negSpline= np.zeros((max_y,max_x))
allcellsl10FH= np.zeros((max_y,max_x))
allcellsl10RH= np.zeros((max_y,max_x))
allcellsl10FHSpl= np.zeros((max_y,max_x))
allcellsl10RHSpl= np.zeros((max_y,max_x))
BothSplines10 = np.zeros((max_y,max_x))
BothSplinesRed10 = np.zeros((max_y,max_x))
BothFin = np.zeros((max_y,max_x))

allcellsl11pos = np.zeros((max_y,max_x))
allcellsl11posT= np.zeros((max_y,max_x))
allcellsl11posSpline= np.zeros((max_y,max_x))
allcellsl11neg = np.zeros((max_y,max_x))
allcellsl11negT= np.zeros((max_y,max_x))
allcellsl11negSpline= np.zeros((max_y,max_x))
allcellsl11FH= np.zeros((max_y,max_x))
allcellsl11RH= np.zeros((max_y,max_x))
allcellsl11FHSpl= np.zeros((max_y,max_x))
allcellsl11RHSpl= np.zeros((max_y,max_x))

allcellsl12pos = np.zeros((max_y,max_x))
allcellsl12posT= np.zeros((max_y,max_x))
allcellsl12posSpline= np.zeros((max_y,max_x))
allcellsl12neg = np.zeros((max_y,max_x))
allcellsl12negT= np.zeros((max_y,max_x))
allcellsl12negSpline= np.zeros((max_y,max_x))
allcellsl12FH= np.zeros((max_y,max_x))
allcellsl12RH= np.zeros((max_y,max_x))
allcellsl12FHSpl= np.zeros((max_y,max_x))
allcellsl12RHSpl= np.zeros((max_y,max_x))


allcellsPosl1Spline = np.zeros((max_y,max_x))
allcellsPosl2Spline = np.zeros((max_y,max_x))
allcellsPosl3Spline = np.zeros((max_y,max_x))
allcellsPosl4Spline = np.zeros((max_y,max_x))
allcellsPosl5Spline = np.zeros((max_y,max_x))
allcellsPosl6Spline = np.zeros((max_y,max_x))
allcellsPosl7Spline = np.zeros((max_y,max_x))
allcellsPosl8Spline = np.zeros((max_y,max_x))
allcellsPosl9Spline = np.zeros((max_y,max_x))
allcellsPosl10Spline = np.zeros((max_y,max_x))
allcellsPosl11Spline = np.zeros((max_y,max_x))

allcellsNormalizedl1 = np.zeros((max_y,max_x))
allcellsNormalizedl2 = np.zeros((max_y,max_x))
allcellsNormalizedl3 = np.zeros((max_y,max_x))
allcellsNormalizedl4 = np.zeros((max_y,max_x))
allcellsNormalizedl5 = np.zeros((max_y,max_x))
allcellsNormalizedl6 = np.zeros((max_y,max_x))
allcellsNormalizedl7 = np.zeros((max_y,max_x))
allcellsNormalizedl8 = np.zeros((max_y,max_x))
allcellsNormalizedl9 = np.zeros((max_y,max_x))
allcellsNormalizedl10 = np.zeros((max_y,max_x))
allcellsNormalizedl11 = np.zeros((max_y,max_x))


for fi in files: 
	#print "new round"

	pfile = open(parameterfileIn +"/" + fi,'r')
	imfile = open(imagefileIn+ "/" +fi,'r')

	#print "pfile: ", pfile
	#print "imfile: ", imfile
	#pdb.set_trace()
	All_Ellipses=[]
	ellipseDict = {}


	count = 0
	icount = 0

	pfile.readline()
	imfile.readline()
	sline = imfile.readline()
	
	arrafilledboth = np.zeros((max_y,max_x))
	
	allcellsPosNormed = np.zeros((max_y,max_x))
	allcellsPosSpline = np.zeros((max_y,max_x))
	allcellsSplBothPos = np.zeros((max_y,max_x))
	
	allcellsSplinenoch = np.zeros((max_y,max_x))
	allcellsNormed = np.zeros((max_y,max_x))
	allcellsReduced = np.zeros((max_y,max_x))
	
	allarraydoubleReduced = np.zeros((max_y,max_x))
	

	for line in pfile: 
		if line.startswith('[') or line.startswith('0.') or line.startswith('1.') or line.startswith(' '): 
			continue
		#print "line: ", line
		#print "sline1: ",sline

		fline = line.split("\t")
		ratio = float(fline[7])
		a = float(fline[1])
		b = float(fline[2])
		a2 = float(fline[3])
		b2 = float(fline[4])
		area = float(fline[5])
		perimeter = float(fline[6])
		length = float(fline[8])
		x0 = float(fline[9])
		y0 = float(fline[10])
		x02 = float(fline[11])
		y02 = float(fline[12])
		theta1 = float(fline[13])
		theta2 = float(fline[14])
		
		#print area
		if area<500: 
			#print "imfile: ", imfile, "pfile: ", pfile
			imfile.readline()
			sline = imfile.readline()
			continue
		#print "not continued"
		#print icount
		aMikro = a*0.13
		bMikro = b*0.13
		a2Mikro = a2*0.13
		b2Mikro = b2*0.13
		areaMikro = area * 0.13
		perimeterMikro = perimeter * 0.13
		lengthMikro = length * 0.13
		x0Mikro = x0 * 0.13
		y0Mikro = y0 * 0.13


		if minlength < length <= l1: 

			#print "l1", l1, pfile, fline[0]
			a_averagelistl1.append(a)
			a2_averagelistl1.append(a2)
			b_averagelistl1.append(b)
			b2_averagelistl1.append(b2)
			length_averagelistl1.append(length)
			area_averagelistl1.append(area)
			perimeter_averagelistl1.append(perimeter)
			aMikro_averagelistl1.append(aMikro)
			bMikro_averagelistl1.append(bMikro)
			a2Mikro_averagelistl1.append(a2Mikro)
			b2Mikro_averagelistl1.append(b2Mikro)
			areaMikro_averagelistl1.append(areaMikro)
			perimeterMikro_averagelistl1.append(perimeterMikro)
			lengthMikro_averagelistl1.append(lengthMikro)
			x0_averagelistl1.append(x0)
			y0_averagelistl1.append(y0)
			x0Mikro_averagelistl1.append(x0Mikro)
			y0Mikro_averagelistl1.append(y0Mikro)


		elif l1 < length <= l2: 


			a_averagelistl2.append(a)
			a2_averagelistl2.append(a2)
			b_averagelistl2.append(b)
			b2_averagelistl2.append(b2)
			length_averagelistl2.append(length)
			area_averagelistl2.append(area)
			perimeter_averagelistl2.append(perimeter)
			aMikro_averagelistl2.append(aMikro)
			bMikro_averagelistl2.append(bMikro)
			a2Mikro_averagelistl2.append(a2Mikro)
			b2Mikro_averagelistl2.append(b2Mikro)
			areaMikro_averagelistl2.append(areaMikro)
			perimeterMikro_averagelistl2.append(perimeterMikro)
			lengthMikro_averagelistl2.append(lengthMikro)
			x0_averagelistl2.append(x0)
			y0_averagelistl2.append(y0)
			x0Mikro_averagelistl2.append(x0Mikro)
			y0Mikro_averagelistl2.append(y0Mikro)

		elif l2 < length <= l3: 


			a_averagelistl3.append(a)
			a2_averagelistl3.append(a2)
			b_averagelistl3.append(b)
			b2_averagelistl3.append(b2)
			length_averagelistl3.append(length)
			area_averagelistl3.append(area)
			perimeter_averagelistl3.append(perimeter)
			aMikro_averagelistl3.append(aMikro)
			bMikro_averagelistl3.append(bMikro)
			a2Mikro_averagelistl3.append(a2Mikro)
			b2Mikro_averagelistl3.append(b2Mikro)
			areaMikro_averagelistl3.append(areaMikro)
			perimeterMikro_averagelistl3.append(perimeterMikro)
			lengthMikro_averagelistl3.append(lengthMikro)
			x0_averagelistl3.append(x0)
			y0_averagelistl3.append(y0)
			x0Mikro_averagelistl3.append(x0Mikro)
			y0Mikro_averagelistl3.append(y0Mikro)


		elif l3 < length <= l4: 


			a_averagelistl4.append(a)
			a2_averagelistl4.append(a2)
			b_averagelistl4.append(b)
			b2_averagelistl4.append(b2)
			length_averagelistl4.append(length)
			area_averagelistl4.append(area)
			perimeter_averagelistl4.append(perimeter)
			aMikro_averagelistl4.append(aMikro)
			bMikro_averagelistl4.append(bMikro)
			a2Mikro_averagelistl4.append(a2Mikro)
			b2Mikro_averagelistl4.append(b2Mikro)
			areaMikro_averagelistl4.append(areaMikro)
			perimeterMikro_averagelistl4.append(perimeterMikro)
			lengthMikro_averagelistl4.append(lengthMikro)
			x0_averagelistl4.append(x0)
			y0_averagelistl4.append(y0)
			x0Mikro_averagelistl4.append(x0Mikro)
			y0Mikro_averagelistl4.append(y0Mikro)

		elif l4 < length <= l5: 


			a_averagelistl5.append(a)
			a2_averagelistl5.append(a2)
			b_averagelistl5.append(b)
			b2_averagelistl5.append(b2)
			length_averagelistl5.append(length)
			area_averagelistl5.append(area)
			perimeter_averagelistl5.append(perimeter)
			aMikro_averagelistl5.append(aMikro)
			bMikro_averagelistl5.append(bMikro)
			a2Mikro_averagelistl5.append(a2Mikro)
			b2Mikro_averagelistl5.append(b2Mikro)
			areaMikro_averagelistl5.append(areaMikro)
			perimeterMikro_averagelistl5.append(perimeterMikro)
			lengthMikro_averagelistl5.append(lengthMikro)
			x0_averagelistl5.append(x0)
			y0_averagelistl5.append(y0)
			x0Mikro_averagelistl5.append(x0Mikro)
			y0Mikro_averagelistl5.append(y0Mikro)


		elif l5 < length <= l6: 


			a_averagelistl6.append(a)
			a2_averagelistl6.append(a2)
			b_averagelistl6.append(b)
			b2_averagelistl6.append(b2)
			length_averagelistl6.append(length)
			area_averagelistl6.append(area)
			perimeter_averagelistl6.append(perimeter)
			aMikro_averagelistl6.append(aMikro)
			bMikro_averagelistl6.append(bMikro)
			a2Mikro_averagelistl6.append(a2Mikro)
			b2Mikro_averagelistl6.append(b2Mikro)
			areaMikro_averagelistl6.append(areaMikro)
			perimeterMikro_averagelistl6.append(perimeterMikro)
			lengthMikro_averagelistl6.append(lengthMikro)
			x0_averagelistl6.append(x0)
			y0_averagelistl6.append(y0)
			x0Mikro_averagelistl6.append(x0Mikro)
			y0Mikro_averagelistl6.append(y0Mikro)


		elif l6 < length <= l7: 


			a_averagelistl7.append(a)
			a2_averagelistl7.append(a2)
			b_averagelistl7.append(b)
			b2_averagelistl7.append(b2)
			length_averagelistl7.append(length)
			area_averagelistl7.append(area)
			perimeter_averagelistl7.append(perimeter)
			aMikro_averagelistl7.append(aMikro)
			bMikro_averagelistl7.append(bMikro)
			a2Mikro_averagelistl7.append(a2Mikro)
			b2Mikro_averagelistl7.append(b2Mikro)
			areaMikro_averagelistl7.append(areaMikro)
			perimeterMikro_averagelistl7.append(perimeterMikro)
			lengthMikro_averagelistl7.append(lengthMikro)
			x0_averagelistl7.append(x0)
			y0_averagelistl7.append(y0)
			x0Mikro_averagelistl7.append(x0Mikro)
			y0Mikro_averagelistl7.append(y0Mikro)

		elif l7 < length <= l8: 


			a_averagelistl8.append(a)
			a2_averagelistl8.append(a2)
			b_averagelistl8.append(b)
			b2_averagelistl8.append(b2)
			length_averagelistl8.append(length)
			area_averagelistl8.append(area)
			perimeter_averagelistl8.append(perimeter)
			aMikro_averagelistl8.append(aMikro)
			bMikro_averagelistl8.append(bMikro)
			a2Mikro_averagelistl8.append(a2Mikro)
			b2Mikro_averagelistl8.append(b2Mikro)
			areaMikro_averagelistl8.append(areaMikro)
			perimeterMikro_averagelistl8.append(perimeterMikro)
			lengthMikro_averagelistl8.append(lengthMikro)
			x0_averagelistl8.append(x0)
			y0_averagelistl8.append(y0)
			x0Mikro_averagelistl8.append(x0Mikro)
			y0Mikro_averagelistl8.append(y0Mikro)

		elif l8 < length <= l9: 


			a_averagelistl9.append(a)
			a2_averagelistl9.append(a2)
			b_averagelistl9.append(b)
			b2_averagelistl9.append(b2)
			length_averagelistl9.append(length)
			area_averagelistl9.append(area)
			perimeter_averagelistl9.append(perimeter)
			aMikro_averagelistl9.append(aMikro)
			bMikro_averagelistl9.append(bMikro)
			a2Mikro_averagelistl9.append(a2Mikro)
			b2Mikro_averagelistl9.append(b2Mikro)
			areaMikro_averagelistl9.append(areaMikro)
			perimeterMikro_averagelistl9.append(perimeterMikro)
			lengthMikro_averagelistl9.append(lengthMikro)
			x0_averagelistl9.append(x0)
			y0_averagelistl9.append(y0)
			x0Mikro_averagelistl9.append(x0Mikro)
			y0Mikro_averagelistl9.append(y0Mikro)

		elif l9 < length <= l10: 


			a_averagelistl10.append(a)
			a2_averagelistl10.append(a2)
			b_averagelistl10.append(b)
			b2_averagelistl10.append(b2)
			length_averagelistl10.append(length)
			area_averagelistl10.append(area)
			perimeter_averagelistl10.append(perimeter)
			aMikro_averagelistl10.append(aMikro)
			bMikro_averagelistl10.append(bMikro)
			a2Mikro_averagelistl10.append(a2Mikro)
			b2Mikro_averagelistl10.append(b2Mikro)
			areaMikro_averagelistl10.append(areaMikro)
			perimeterMikro_averagelistl10.append(perimeterMikro)
			lengthMikro_averagelistl10.append(lengthMikro)
			x0_averagelistl10.append(x0)
			y0_averagelistl10.append(y0)
			x0Mikro_averagelistl10.append(x0Mikro)
			y0Mikro_averagelistl10.append(y0Mikro)

		elif l10 < length <= l11: 


			a_averagelistl11.append(a)
			a2_averagelistl11.append(a2)
			b_averagelistl11.append(b)
			b2_averagelistl11.append(b2)
			length_averagelistl11.append(length)
			area_averagelistl11.append(area)
			perimeter_averagelistl11.append(perimeter)
			aMikro_averagelistl11.append(aMikro)
			bMikro_averagelistl11.append(bMikro)
			a2Mikro_averagelistl11.append(a2Mikro)
			b2Mikro_averagelistl11.append(b2Mikro)
			areaMikro_averagelistl11.append(areaMikro)
			perimeterMikro_averagelistl11.append(perimeterMikro)
			lengthMikro_averagelistl11.append(lengthMikro)
			x0_averagelistl11.append(x0)
			y0_averagelistl11.append(y0)
			x0Mikro_averagelistl11.append(x0Mikro)
			y0Mikro_averagelistl11.append(y0Mikro)

		elif l11 < length <= l12: 


			a_averagelistl12.append(a)
			a2_averagelistl12.append(a2)
			b_averagelistl12.append(b)
			b2_averagelistl12.append(b2)
			length_averagelistl12.append(length)
			area_averagelistl12.append(area)
			perimeter_averagelistl12.append(perimeter)
			aMikro_averagelistl12.append(aMikro)
			bMikro_averagelistl12.append(bMikro)
			a2Mikro_averagelistl12.append(a2Mikro)
			b2Mikro_averagelistl12.append(b2Mikro)
			areaMikro_averagelistl12.append(areaMikro)
			perimeterMikro_averagelistl12.append(perimeterMikro)
			lengthMikro_averagelistl12.append(lengthMikro)
			x0_averagelistl12.append(x0)
			y0_averagelistl12.append(y0)
			x0Mikro_averagelistl12.append(x0Mikro)
			y0Mikro_averagelistl12.append(y0Mikro)


		
		

		#x0_c = max_x/3
		x0_c = max_x/2


		if x02>x0: 
			x02_c = max_x/2+(x02-x0)
		else: 
			x02_c = max_x/2-(x0-x02)

		#y0_c = max_y/3
		y0_c = max_y/2

		if y02>y0:
			y02_c = max_y/2+(y02-y0)
		else: 
			y02_c = max_y/2-(y0-y02)

		el1=makeEllipse1(x0_c,y0_c,a,b,theta1)
		el2=makeEllipse1(x02_c,y02_c,a2,b2,theta2)
		Ar1 = makeArray(el1,max_x,max_y)
		Ar2 = makeArray(el2,max_x,max_y)
		both = Ar1+Ar2
		#MakeImage(both, "")
		#ShowImage()


		#line through ellipse centers
		###############################
		xli = np.linspace(0,max_x-1,max_x)
		yli = np.linspace(y0_c,y0_c, max_x)
		x_ar=np.array([xli])
		y_ar=np.array([yli])

		horizontal_line = np.append(x_ar,y_ar, axis = 0)
		
		if x02_c == x0_c: 
			#print "len5if"
			xline = np.linspace(0,max_x/3-1,max_x/3)
			yline = np.linspace(y0_c,y0_c, max_x/3)
			x_array=np.array([xline])
			y_array=np.array([yline])
			array_line = np.append(x_array,y_array, axis = 0)
			
		else:
			m = (y02_c-y0_c)/(x02_c-x0_c)
			xline=np.linspace(0,max_x/3-1,max_x/3)	
			yline = np.around(m*(xline-x0_c)+y0_c)
			for yl in range(len(yline)): 
				if yline[yl] >= max_y: 
					yline[yl]=max_y-1
				elif yline[yl] < 0: 
					yline[yl] = 0
			x_array=np.array([xline])
			y_array=np.array([yline])
			array_line = np.append(x_array,y_array, axis = 0)

		ua = makeArray(horizontal_line,max_x,max_y)
		va = makeArray(array_line,max_x,max_y)
		both = Ar1+Ar2+ua+va
		
		u = (horizontal_line[0][5]-horizontal_line[0][0],horizontal_line[1][5]-horizontal_line[1][0])
		#v = (array_line[0][5]-array_line[0][0],array_line[1][5]-array_line[1][0])
		v = (x02_c-x0_c,y02_c-y0_c)
		theta = calculate(u,v)
		oldtheta = theta

		if x02>x0 and y02>y0: 
			theta =-theta
		elif x02<x0 and y02>y0: 
			theta=-theta
		
		#print "new angle: ", theta
		#print "old theta: ", oldtheta
		eltheta1 =theta+theta1
		eltheta2 = theta+theta2
		l = np.sqrt((y02_c-y0_c)**2+(x02_c-x0_c)**2)
		El1 = makeEllipse1(x0_c,y0_c,a,b,eltheta1)
		El2 = makeEllipse1(x0_c+l,y0_c,a2,b2,eltheta2)

		arra1=makeArray(El1,max_x,max_y)
		arra2 = makeArray(El2,max_x,max_y)
		barray = arra1+arra2



		arra1filled = ndimage.binary_fill_holes(arra1).astype(int)

		arra2filled = ndimage.binary_fill_holes(arra2).astype(int)

		arrafilledboth = arra1filled + arra2filled
		

		for h in range(np.size(arrafilledboth,0)):
			for u in range(np.size(arrafilledboth,1)):
				if arrafilledboth[h][u] > 1: 
					barray[h][u] = 0

		if minlength < length <= l1:
			allarrayl1 = allarrayl1 + barray
			allarrayreducedl1 = allarrayreducedl1 + barray

		elif l1 < length <= l2:
			allarrayl2 = allarrayl2 + barray
			allarrayreducedl2 = allarrayreducedl2 + barray

		elif l2 < length <= l3:
			allarrayl3 = allarrayl3 + barray
			allarrayreducedl3 = allarrayreducedl3 + barray

		elif l3 < length <= l4:
			allarrayl4 = allarrayl4 + barray
			allarrayreducedl4 = allarrayreducedl4 + barray

		elif l4 < length <= l5:
			allarrayl5 = allarrayl5 + barray
			allarrayreducedl5 = allarrayreducedl5 + barray

		elif l5 < length <= l6:
			allarrayl6 = allarrayl6 + barray
			allarrayreducedl6 = allarrayreducedl6 + barray
			#MakeImagetrMicro(allarrayl6, '', x0_c, y0_c)
			#plt.show()

		elif l6 < length <= l7:
			allarrayl7 = allarrayl7 + barray
			allarrayreducedl7 = allarrayreducedl7 + barray

		elif l7 < length <= l8:
			allarrayl8 = allarrayl8 + barray
			allarrayreducedl8 = allarrayreducedl8 + barray

		elif l8 < length <= l9:
			allarrayl9 = allarrayl9 + barray
			allarrayreducedl9 = allarrayreducedl9 + barray

		elif l9 < length <= l10:
			allarrayl10 = allarrayl10 + barray
			allarrayreducedl10 = allarrayreducedl10 + barray
			#MakeImagetr(allarrayl10,'',x0_c,y0_c)

			#plt.show()
			#MakeImagetr(allarrayl10,'',x0_c,y0_c)
			#plt.show()

		elif l10 < length <= l11:
			allarrayl11 = allarrayl11 + barray
			allarrayreducedl11 = allarrayreducedl11 + barray

		elif l11 < length <= l12:
			allarrayl12 = allarrayl12 + barray
			allarrayreducedl12 = allarrayreducedl12 + barray


		A = []
	 	matrix =sline
	 	tuple_rx = re.compile("\(\s*(\d+),\s*(\d+)\)")
	 	for match in tuple_rx.finditer(matrix): 
	 		A.append((int(match.group(1)),int(match.group(2))))

	 	c = []
	 	d = []

	 	for i in A: 
			c.append(i[0])
			d.append(i[1])

		A=(c,d)
		#print "len: ", len(A[0])
		#print A
		if len(A[0]) == 0: 
			#print "EMPTY"
			continue
		#print "NOT EMPTY"
		#print "sline: ", sline
		#print"c,d: ", c, d
		#A=sorted(A)
		A=np.array(A)
		imAr = makeArray(A,max_x,max_y)
		#MakeImage(imAr, "")
		#ShowImage()
		#a=40
		#b=10
		#a2 = 20
		#b2 = 5
		#x0=40
		#y0=10
		#x02=70
		#y02=10
		el1=makeEllipse1(x0,y0,a,b,0)
		el2=makeEllipse1(x02,y02,a2,b2,0)

		#pl.axis([-100,100,-100,100])
		#pl.plot(A[0,:],A[1,:],'ro')
		#pl.show()

		for i in range(np.size(A[0])): #damit um mittelpunkt rotiert wird
		
			A[0][i] = A[0][i]-x0
			A[1][i] = A[1][i]-y0

		for i in range(np.size(el1[0])):
			el1[0][i] = el1[0][i]-x0
			el1[1][i] = el1[1][i]-y0
			el2[0][i] = el2[0][i]-x0
			el2[1][i] = el2[1][i]-y0

		#pl.axis([-100,100,-100,100])
		#pl.plot(el1[0,:],el1[1,:])
		#pl.plot(el2[0,:],el2[1,:])
		#pl.show()
		
		#pl.axis([-100,100,-100,100])
		#pl.plot(A[0,:],A[1,:])
		#pl.show()
		midp = [[x0_c],[y0_c]]
		midp = np.array(midp)
		#pl.axis([-100,100,-100,100])
		#pl.plot(A[0,:],A[1,:],el1[0,:],el1[1,:],midp[0,:],midp[1,:],'ro')
		#pl.show()

		RotEl = rotate(el1,theta)
		RotEl2 = rotate(el2,theta)

	 	RotA = rotate(A,theta)
	 	RotAPos = rotate(A,theta)
	 	
	 	RotAmaxx = np.amax(RotA[0])
	 	RotAmaxy = np.amax(RotA[1])
	###########################################hier!!!!!!!!#########################
	  	RotAFH = rotate(A,theta)
	  	RotASH = rotate(A,theta)
	  	#print "xo: ", x0, "xo_c: ", x0_c, "y0: ", y0, "y0_c: ", y0_c
	  	for i in range(np.size(el1[0])):
			el1[0][i] = el1[0][i]+x0_c
			el1[1][i] = el1[1][i]+y0_c
			el2[0][i] = el2[0][i]+x0_c
			el2[1][i] = el2[1][i]+y0_c

	 # 	RotASH = RotA
	 # 	print np.size(A[0])
	 # 	print np.size(RotA[0])
	 # 	print RotA
	 # 	print range(np.size(A[0]))

		# print RotA[0][0]
		for ir in range(np.size(A[0])):
		# 	print a
			if RotA[0][ir]>0:
				RotAFH[0][ir] = RotA[0][ir]
				RotAFH[1][ir] = RotA[1][ir]
		 		RotASH[0][ir] = 0
		 		RotASH[1][ir] = 0
		 	else:
		 		RotASH[0][ir] = RotA[0][ir]
		 		RotASH[1][ir] = RotA[1][ir]
		 		RotAFH[0][ir] = 0
		 		RotAFH[1][ir] = 0

		#pl.axis([-100,100,-100,100])
		#pl.plot(RotA[0,:],RotA[1,:])
		#pl.show()

		#pl.axis([-100,100,-100,100])
		#pl.plot(RotAFH[0,:],RotAFH[1,:])
		#pl.show()

		#pl.axis([-100,100,-100,100])
		#pl.plot(RotA[0,:],RotA[1,:],midp[0,:],midp[1,:],'ro')
		#pl.show()

		zarray = makeArray(RotA,max_x,max_y)
		#MakeImage(zarray, '')
		#plt.show()

	 	for i in range(np.size(A[0])): 
	 	
	 		if RotA[1][i] > 0:
	 			RotAPos[1][i] = -RotAPos[1][i] 
			RotA[0][i] = RotA[0][i]+x0_c
			RotA[1][i] = RotA[1][i]+y0_c
			RotAPos[0][i] = RotAPos[0][i]+x0_c
			RotAPos[1][i] = RotAPos[1][i]+y0_c
			RotAFH[0][i] = RotAFH[0][i]+x0_c
			RotAFH[1][i] = RotAFH[1][i] + y0_c
			RotASH[0][i] = RotASH[0][i]+x0_c
			RotASH[1][i] = RotASH[1][i] + y0_c
		
		RotAminy = np.amin(RotA[1])
	 	RotAminx = np.amin(RotA[0])
	 	RotAmaxx = np.amax(RotA[1])
	 	RotAmaxy = np.amax(RotA[0])
	 	#print "area: ", area, "icount: ", icount
	 	#print "maxx: ", RotAmaxx, "maxy: ", RotAmaxy


		
		if RotAminx < 0: 
			#print "RotAminx kleiner 0"
			RotA[0] = RotA[0]-RotAminx
			RotAPos[0] = RotAPos[0]-RotAminx
	 	if RotAminy < 0: 
	 		#print "RotAminy kleiner 0"
	 		RotA[1] = RotA[1]-RotAminy
	 		RotAPos[1] = RotAPos[1]-RotAminy
	 	ellipse1Array = makeArray(el1, max_x, max_y)
		#print "RotA: ", RotA, length,area
	 	imageArray = makeArray(RotA,max_x,max_y)
	 	imageArrayPos = makeArray(RotAPos,max_x,max_y)

	 	imageArrayFH = makeArray(RotAFH,max_x,max_y)
	 	imageArraySH = makeArray(RotASH,max_x,max_y)
	 	#print x0,y0,x0_c,y0_c
	 	
	 	MidArray = makeArray(midp,max_x,max_y)

	 	#pl.axis([-100,100,-100,100])
		#pl.plot(RotA[0,:],RotA[1,:],RotEl[0,:],RotEl[1,:],midp[0,:],midp[1,:],'ro')
		#pl.show()
		#MakeImage(MidArray,'')
		#plt.show()
	 	#MakeImage(imageArray,'')
	 	#plt.show()
	 	#MakeImage2(imageArray,MidArray)
	 	#plt.show()


	 	if minlength < length <= l1: 

	 		#print "l1: ",length
		 	allcellsl1 = allcellsl1 + imageArray
		 	allcellsPosl1 = allcellsPosl1 + imageArrayPos
		 	allcellsFHl1 = allcellsFHl1 + imageArrayFH
		 	allcellsSHl1 = allcellsSHl1 + imageArraySH
			#MakeImage(imageArray, "test")
			#plt.show()

		elif l1 < length <= l2: 

		 	allcellsl2 = allcellsl2 + imageArray
		 	allcellsPosl2 = allcellsPosl2 + imageArrayPos
		 	allcellsFHl2 = allcellsFHl2 + imageArrayFH
		 	allcellsSHl2 = allcellsSHl2 + imageArraySH

		elif l2 < length <= l3: 

		 	allcellsl3 = allcellsl3 + imageArray
		 	allcellsPosl3 = allcellsPosl3 + imageArrayPos
		 	allcellsFHl3 = allcellsFHl3 + imageArrayFH
		 	allcellsSHl3 = allcellsSHl3 + imageArraySH

		elif l3 < length <= l4: 

		 	allcellsl4 = allcellsl4 + imageArray
		 	allcellsPosl4 = allcellsPosl4 + imageArrayPos
		 	allcellsFHl4 = allcellsFHl4 + imageArrayFH
		 	allcellsSHl4 = allcellsSHl4 + imageArraySH

		elif l4 < length <= l5: 

		 	allcellsl5 = allcellsl5 + imageArray
		 	allcellsPosl5 = allcellsPosl5 + imageArrayPos
		 	allcellsFHl5 = allcellsFHl5 + imageArrayFH
		 	allcellsSHl5 = allcellsSHl5 + imageArraySH

		 	


		elif l5 < length <= l6: 
			allcellsl6 = allcellsl6 + imageArray
			allcellsPosl6 = allcellsPosl6 + imageArrayPos
			allcellsFHl6 = allcellsFHl6 + imageArrayFH
			allcellsSHl6 = allcellsSHl6 + imageArraySH
			


		elif l6 < length <= l7: 

		 	allcellsl7 = allcellsl7 + imageArray
		 	allcellsPosl7 = allcellsPosl7 + imageArrayPos
		 	allcellsFHl7 = allcellsFHl7 + imageArrayFH
		 	allcellsSHl7 = allcellsSHl7 + imageArraySH

		elif l7 < length <= l8: 

		 	allcellsl8 = allcellsl8 + imageArray
		 	allcellsPosl8 = allcellsPosl8 + imageArrayPos
		 	allcellsFHl8 = allcellsFHl8 + imageArrayFH
		 	allcellsSHl8 = allcellsSHl8 + imageArraySH


		elif l8 < length <= l9: 

		 	allcellsl9 = allcellsl9 + imageArray
		 	allcellsPosl9 = allcellsPosl9 + imageArrayPos
		 	allcellsFHl9 = allcellsFHl9 + imageArrayFH
		 	allcellsSHl9 = allcellsSHl9 + imageArraySH
		 	#MakeImage(imageArray,'')
	 		#plt.show()
	 		#MakeImage2(imageArray,MidArray,ellipse1Array)
	 		#plt.show()
	 		#MakeImagetr(allcellsl9, MidArray,'', x0_c, y0_c)
		 	#plt.show()


		elif l9 < length <= l10: 

		 	allcellsl10 = allcellsl10 + imageArray
		 	allcellsPosl10 = allcellsPosl10 + imageArrayPos
		 	allcellsFHl10 = allcellsFHl10 + imageArrayFH
		 	allcellsSHl10 = allcellsSHl10 + imageArraySH


		elif l10 < length <= l11: 

		 	allcellsl11 = allcellsl11 + imageArray
		 	allcellsPosl11 = allcellsPosl11 + imageArrayPos
		 	allcellsFHl11 = allcellsFHl11 + imageArrayFH
		 	allcellsSHl11 = allcellsSHl11 + imageArraySH


		elif l11 < length <= l12: 

		 	allcellsl12 = allcellsl12 + imageArray
		 	allcellsPosl12 = allcellsPosl12 + imageArrayPos
		 	allcellsFHl12 = allcellsFHl12 + imageArrayFH
		 	allcellsSHl12 = allcellsSHl12 + imageArraySH

		icount += 1
		imfile.readline()
		sline = imfile.readline()

x0av1 = np.mean(x0_averagelistl1)
y0av1 = np.mean(y0_averagelistl1)

x0av2 = np.mean(x0_averagelistl2)
y0av2 = np.mean(y0_averagelistl2)

x0av3 = np.mean(x0_averagelistl3)
y0av3 = np.mean(y0_averagelistl3)

x0av4 = np.mean(x0_averagelistl4)
y0av4 = np.mean(y0_averagelistl4)

x0av5 = np.mean(x0_averagelistl5)
y0av5 = np.mean(y0_averagelistl5)

x0av6 = np.mean(x0_averagelistl6)
y0av6 = np.mean(y0_averagelistl6)

x0av7 = np.mean(x0_averagelistl7)
y0av7 = np.mean(y0_averagelistl7)

x0av8 = np.mean(x0_averagelistl8)
y0av8 = np.mean(y0_averagelistl8)

x0av9 = np.mean(x0_averagelistl9)
y0av9 = np.mean(y0_averagelistl9)

x0av10 = np.mean(x0_averagelistl10)
y0av10 = np.mean(y0_averagelistl10)

x0av11 = np.mean(x0_averagelistl11)
y0av11 = np.mean(y0_averagelistl11)

x0av12 = np.mean(x0_averagelistl12)
y0av12 = np.mean(y0_averagelistl12)

minlength = format(minlength,'.1f')
l1 = format(l1,'.1f')
l2 =format(l2,'.1f')
l3 = format(l3,'.1f')
l4 = format(l4,'.1f')
l5 =format(l5,'.1f')
l6 = format(l6,'.1f')
l7 =format(l7,'.1f')
l8 = format(l8,'.1f')
l9 = format(l9,'.1f')
l10 =format(l10,'.1f')
l11 = format(l11,'.1f')
l12 =format(l12,'.1f')

minlengthMicro = format(minlengthMicro,'.1f')
l1Micro = format(l1Micro,'.1f')
l2Micro =format(l2Micro,'.1f')
l3Micro = format(l3Micro,'.1f')
l4Micro = format(l4Micro,'.1f')
l5Micro =format(l5Micro,'.1f')
l6Micro = format(l6Micro,'.1f')
l7Micro =format(l7Micro,'.1f')
l8Micro = format(l8Micro,'.1f')
l9Micro = format(l9Micro,'.1f')
l10Micro =format(l10Micro,'.1f')
l11Micro = format(l11Micro,'.1f')
l12Micro =format(l12Micro,'.1f')

maxvl1 = np.amax(allcellsl1)
maxvl2 = np.amax(allcellsl2)
maxvl3 = np.amax(allcellsl3)
maxvl4 = np.amax(allcellsl4)
maxvl5 = np.amax(allcellsl5)
maxvl6 = np.amax(allcellsl6)
maxvl7 = np.amax(allcellsl7)
maxvl8 = np.amax(allcellsl8)
maxvl9 = np.amax(allcellsl9)
maxvl10 = np.amax(allcellsl10)
maxvl11 = np.amax(allcellsl11)

for row in range(np.size(allcellsl1,0)):
	for color in range(np.size(allcellsl1,1)): 
		allcellsNormalizedl1[row][color] = allcellsl1[row][color]/maxvl1

for row in range(np.size(allcellsl2,0)):
	for color in range(np.size(allcellsl2,1)): 
		allcellsNormalizedl2[row][color] = allcellsl2[row][color]/maxvl2

for row in range(np.size(allcellsl3,0)):
	for color in range(np.size(allcellsl3,1)): 
		allcellsNormalizedl3[row][color] = allcellsl3[row][color]/maxvl3

for row in range(np.size(allcellsl4,0)):
	for color in range(np.size(allcellsl4,1)): 
		allcellsNormalizedl4[row][color] = allcellsl4[row][color]/maxvl4

for row in range(np.size(allcellsl5,0)):
	for color in range(np.size(allcellsl5,1)): 
		allcellsNormalizedl5[row][color] = allcellsl5[row][color]/maxvl5

for row in range(np.size(allcellsl6,0)):
	for color in range(np.size(allcellsl6,1)): 
		allcellsNormalizedl6[row][color] = allcellsl6[row][color]/maxvl6

for row in range(np.size(allcellsl7,0)):
	for color in range(np.size(allcellsl7,1)): 
		allcellsNormalizedl7[row][color] = allcellsl7[row][color]/maxvl7

for row in range(np.size(allcellsl8,0)):
	for color in range(np.size(allcellsl8,1)): 
		allcellsNormalizedl8[row][color] = allcellsl8[row][color]/maxvl8

for row in range(np.size(allcellsl9,0)):
	for color in range(np.size(allcellsl9,1)): 
		allcellsNormalizedl9[row][color] = allcellsl9[row][color]/maxvl9

for row in range(np.size(allcellsl10,0)):
	for color in range(np.size(allcellsl10,1)): 
		allcellsNormalizedl10[row][color] = allcellsl10[row][color]/maxvl10

for row in range(np.size(allcellsl11,0)):
	for color in range(np.size(allcellsl11,1)): 
		allcellsNormalizedl11[row][color] = allcellsl11[row][color]/maxvl11


MakeNormalizedImage(allcellsNormalizedl1, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(minlength) + " to " + str(l1))

MakeNormalizedImage(allcellsNormalizedl2, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l1Micro) + " to " + str(l2Micro))

MakeNormalizedImage(allcellsNormalizedl3, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l2Micro) + " to " + str(l3Micro))

MakeNormalizedImage(allcellsNormalizedl4, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l3Micro) + " to " + str(l4Micro))

MakeNormalizedImage(allcellsNormalizedl5, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l4Micro) + " to " + str(l5Micro))

MakeNormalizedImage(allcellsNormalizedl6, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l5Micro) + " to " + str(l6Micro))

MakeNormalizedImage(allcellsNormalizedl7, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l6Micro) + " to " + str(l7Micro))

MakeNormalizedImage(allcellsNormalizedl8, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l7Micro) + " to " + str(l8Micro))

MakeNormalizedImage(allcellsNormalizedl9, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l8Micro) + " to " + str(l9Micro))

MakeNormalizedImage(allcellsNormalizedl10, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l9Micro) + " to " + str(l10Micro))

MakeNormalizedImage(allcellsNormalizedl11, '',x0_c-1,y0_c-1)
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/LengthSeries/" + Strain + "/"+str(l10Micro) + " to " + str(l11Micro))

#####ab hier kommentiert

# Cellfile1Norm = open(MatlabfileNorm + "/" + str(minlength) + "_" + str(l1Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl1: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile1Norm.write(cell2+"\n")

# Cellfile2Norm = open(MatlabfileNorm + "/" + str(l1Micro) + "_" + str(l2Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl2: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile2Norm.write(cell2+"\n")

# Cellfile3Norm = open(MatlabfileNorm + "/" + str(l2Micro) + "_" + str(l3Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl3: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile3Norm.write(cell2+"\n")

# Cellfile4Norm = open(MatlabfileNorm + "/" + str(l3Micro) + "_" + str(l4Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl4: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile4Norm.write(cell2+"\n")

# Cellfile5Norm = open(MatlabfileNorm + "/" + str(l4Micro) + "_" + str(l5Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl5: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile5Norm.write(cell2+"\n")

# Cellfile6Norm = open(MatlabfileNorm + "/" + str(l5Micro) + "_" + str(l6Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl6: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile6Norm.write(cell2+"\n")

# Cellfile7Norm = open(MatlabfileNorm + "/" + str(l6Micro) + "_" + str(l7Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl7: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile7Norm.write(cell2+"\n")

# Cellfile8Norm = open(MatlabfileNorm + "/" + str(l7Micro) + "_" + str(l8Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl8: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile8Norm.write(cell2+"\n")

# Cellfile9Norm = open(MatlabfileNorm + "/" + str(l8Micro) + "_" + str(l9Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl9: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile9Norm.write(cell2+"\n")

# Cellfile10Norm = open(MatlabfileNorm + "/" + str(l9Micro) + "_" + str(l10Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl10: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile10Norm.write(cell2+"\n")

# Cellfile11Norm = open(MatlabfileNorm + "/" + str(l10Micro) + "_" + str(l11Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl11: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile11Norm.write(cell2+"\n")

# Cellfile12Norm = open(MatlabfileNorm + "/" + str(l11Micro) + "_" + str(l12Micro) + "_Cells.txt","w")
# for cell in allcellsNormalizedl12: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile12Norm.write(cell2+"\n")

###bis hier kommentiert

# MakeImagetr(allcellsl1, str(minlength) + " pixel to " + str(l1) + " pixel",x0_c-1,y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+str(minlength) + " to " + str(l1) + "_Cells")

# MakeImagetr(allcellsl2,str(l1) + " pixel to " + str(l2) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l1) + " to " + str(l2) + "_Cells")

# MakeImagetr(allcellsl3,str(l2) + " pixel to "+ str(l3) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l2) + " to " + str(l3) + "_Cells")

# MakeImagetr(allcellsl4,str(l3) + " pixel to " + str(l4) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l3) + " to " + str(l4) + "_Cells")

# MakeImagetr(allcellsl5,str(l4) + " pixel to " + str(l5) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l4) + " to " + str(l5) + "_Cells")

# MakeImagetr(allcellsl6,str(l5) + " pixel to " + str(l6) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l5) + " to " + str(l6)+ "_Cells")

# MakeImagetr(allcellsl7,str(l6) +" pixel to "+ str(l7) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l6) + " to " + str(l7) + "_Cells")

# MakeImagetr(allcellsl8,str(l7) +" pixel to " + str(l8) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l7) + " to " + str(l8) + "_Cells")

# MakeImagetr(allcellsl9,str(l8) + " pixel to " + str(l9) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+ str(l8) + " to " + str(l9)+ "_Cells")

# MakeImagetr(allcellsl10,str(l9) +" pixel to " + str(l10) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+ str(l9) + " to " + str(l10) + "_Cells")

# MakeImagetr(allcellsl11,str(l10) + " pixel to " + str(l11) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+str(l10) + " to " + str(l11)+ "_Cells")

# MakeImagetr(allcellsl12,str(l11) + " pixel to "+ str(l12) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l11) + " to " + str(l12) + "_Cells")

# MakeImagetrMicro(allcellsl1,str(minlengthMicro) + " $\mu$m to " + str(l1Micro)+ " $\mu$m",x0_c-1,y0_c-1)
# ShowImage(ImageLocOut +"Micrometer"+ "/" + "_Cells"+str(minlengthMicro) + "_" + str(l1Micro))

# MakeImagetrMicro(allcellsl2,str(l1Micro) + " $\mu$m to "+ str(l2Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer"+ "/" + "_Cells"+str(l1Micro) + "_" + str(l2Micro))

# MakeImagetrMicro(allcellsl3,str(l2Micro) + " $\mu$m to " + str(l3Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer"+ "/" + "_Cells"+str(l2Micro) + "_" + str(l3Micro))

# MakeImagetrMicro(allcellsl4,str(l3Micro) + " $\mu$m to " + str(l4Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer"+  "/" +"_Cells"+str(l3Micro) + "_" + str(l4Micro))

# MakeImagetrMicro(allcellsl5,str(l4Micro) + " $\mu$m to "+ str(l5Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells"+str(l4Micro) + "_" + str(l5Micro))

# MakeImagetrMicro(allcellsl6,str(l5Micro) + " $\mu$m to " + str(l6Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells"+str(l5Micro) + "_" + str(l6Micro))

# MakeImagetrMicro(allcellsl7,str(l6Micro) + " $\mu$m to " + str(l7Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" + "/" +"_Cells"+str(l6Micro) + "_" + str(l7Micro))
 
# MakeImagetrMicro(allcellsl8,str(l7Micro) + " $\mu$m to " + str(l8Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells"+str(l7Micro) + "_" + str(l8Micro))

# MakeImagetrMicro(allcellsl9,str(l8Micro) + " $\mu$m to " + str(l9Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Micrometer" + "/" +"_Cells"+str(l8Micro) + "_" + str(l9Micro))

# MakeImagetrMicro(allcellsl10,str(l9Micro) + " $\mu$m to " + str(l10Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Micrometer" +"/" + "_Cells"+str(l9Micro) + "_" + str(l10Micro))

# MakeImagetrMicro(allcellsl11,str(l10Micro) + " $\mu$m to " + str(l11Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Micrometer" +"/" + "_Cells"+str(l10Micro) + "_" + str(l11Micro))

# MakeImagetrMicro(allcellsl12,str(l11Micro) + " $\mu$m to " + str(l12Micro)+ " $\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells"+str(l11Micro) + "_" + str(l12Micro))


# ###################gespiegelt

# MakeImagetr(allcellsPosl1, str(minlength) + " pixel to "+ str(l1) + " pixel",x0_c-1,y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+str(minlength) + " to " + str(l1) + "_PosCells")

# MakeImagetr(allcellsPosl2,str(l1) + " pixel to "+ str(l2) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l1) + " to " + str(l2) + "_PosCells")

# MakeImagetr(allcellsPosl3,str(l2) + " pixel to " + str(l3) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l2) + " to " + str(l3) + "_PosCells")

# MakeImagetr(allcellsPosl4,str(l3) + " pixel to "+ str(l4) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l3) + " to " + str(l4) + "_PosCells")

# MakeImagetr(allcellsPosl5,str(l4) +" pixel to "+ str(l5) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l4) + " to " + str(l5) + "_PosCells")

# MakeImagetr(allcellsPosl6,str(l5) + " pixel to "+ str(l6) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l5) + " to " + str(l6)+ "_PosCells")

# MakeImagetr(allcellsPosl7,str(l6) + " pixel to " + str(l7) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l6) + " to " + str(l7) + "_PosCells")

# MakeImagetr(allcellsPosl8,str(l7) + " pixel to " + str(l8) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l7) + " to " + str(l8) + "_PosCells")

# MakeImagetr(allcellsPosl9,str(l8) + " pixel to " + str(l9) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+ str(l8) + " to " + str(l9)+ "_PosCells")

# MakeImagetr(allcellsPosl10,str(l9) + " pixel to " + str(l10) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+ str(l9) + " to " + str(l10) + "_PosCells")

# MakeImagetr(allcellsPosl11, str(l10) +" pixel to " + str(l11) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Pixel"+ "/"+str(l10) + " to " + str(l11)+ "_PosCells")

# MakeImagetr(allcellsPosl12, str(l11) + " pixel to "+ str(l12) + " pixel",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Pixel"+ "/"+str(l11) + " to " + str(l12) + "_PosCells")

# MakeImagetrMicro(allcellsPosl1,str(minlengthMicro) + "$\mu$m to " + str(l1Micro)+ "$\mu$m",x0_c-1,y0_c-1)
# ShowImage(ImageLocOut +"Micrometer"+ "/" + "_PosCells"+str(minlengthMicro) + "_" + str(l1Micro))

# MakeImagetrMicro(allcellsPosl2,str(l1Micro) + "$\mu$m to "+ str(l2Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer"+ "/" + "_PosCells"+str(l1Micro) + "_" + str(l2Micro))

# MakeImagetrMicro(allcellsPosl3,str(l2Micro) +"$\mu$m to "+ str(l3Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer"+ "/" + "_PosCells"+str(l2Micro) + "_" + str(l3Micro))

# MakeImagetrMicro(allcellsPosl4,str(l3Micro) + "$\mu$m to " + str(l4Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer"+  "/" +"_PosCells"+str(l3Micro) + "_" + str(l4Micro))

# MakeImagetrMicro(allcellsPosl5,str(l4Micro) +"$\mu$m to " + str(l5Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_PosCells"+str(l4Micro) + "_" + str(l5Micro))

# MakeImagetrMicro(allcellsPosl6,str(l5Micro) + "$\mu$m to " + str(l6Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_PosCells"+str(l5Micro) + "_" + str(l6Micro))

# MakeImagetrMicro(allcellsPosl7,str(l6Micro) + "$\mu$m to " + str(l7Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" + "/" +"_PosCells"+str(l6Micro) + "_" + str(l7Micro))

# MakeImagetrMicro(allcellsPosl8, str(l7Micro) +"$\mu$m to " + str(l8Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_PosCells"+str(l7Micro) + "_" + str(l8Micro))

# MakeImagetrMicro(allcellsPosl9, str(l8Micro) +"$\mu$m to " + str(l9Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Micrometer" + "/" +"_PosCells"+str(l8Micro) + "_" + str(l9Micro))

# MakeImagetrMicro(allcellsPosl10,str(l9Micro) +"$\mu$m to " + str(l10Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Micrometer" +"/" + "_PosCells"+str(l9Micro) + "_" + str(l10Micro))

# MakeImagetrMicro(allcellsPosl11,str(l10Micro) + "$\mu$m to " + str(l11Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut +"Micrometer" +"/" + "_PosCells"+str(l10Micro) + "_" + str(l11Micro))

# MakeImagetrMicro(allcellsPosl12, str(l11Micro) + "$\mu$m to " + str(l12Micro)+ "$\mu$m",x0_c-1, y0_c-1)
# ShowImage(ImageLocOut + "Micrometer" +"/" + "_PosCells"+str(l11Micro) + "_" + str(l12Micro))

# Cellfile1 = open(Matlabfile + "/" + str(minlength) + "_" + str(l1Micro) + "_Cells.txt","w")
# for cell in allcellsl1: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile1.write(cell2+"\n")

# Cellfile2 = open(Matlabfile + "/" + str(l1Micro) + "_" + str(l2Micro) + "_Cells.txt","w")
# for cell in allcellsl2: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile2.write(cell2+"\n")

# Cellfile3 = open(Matlabfile + "/" + str(l2Micro) + "_" + str(l3Micro) + "_Cells.txt","w")
# for cell in allcellsl3: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile3.write(cell2+"\n")

# Cellfile4 = open(Matlabfile + "/" + str(l3Micro) + "_" + str(l4Micro) + "_Cells.txt","w")
# for cell in allcellsl4: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile4.write(cell2+"\n")

# Cellfile5 = open(Matlabfile + "/" + str(l4Micro) + "_" + str(l5Micro) + "_Cells.txt","w")
# for cell in allcellsl5: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile5.write(cell2+"\n")

# Cellfile6 = open(Matlabfile + "/" + str(l5Micro) + "_" + str(l6Micro) + "_Cells.txt","w")
# for cell in allcellsl6: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile6.write(cell2+"\n")

# Cellfile7 = open(Matlabfile + "/" + str(l6Micro) + "_" + str(l7Micro) + "_Cells.txt","w")
# for cell in allcellsl7: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile7.write(cell2+"\n")

# Cellfile8 = open(Matlabfile + "/" + str(l7Micro) + "_" + str(l8Micro) + "_Cells.txt","w")
# for cell in allcellsl8: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile8.write(cell2+"\n")

# Cellfile9 = open(Matlabfile + "/" + str(l8Micro) + "_" + str(l9Micro) + "_Cells.txt","w")
# for cell in allcellsl9: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile9.write(cell2+"\n")

# Cellfile10 = open(Matlabfile + "/" + str(l9Micro) + "_" + str(l10Micro) + "_Cells.txt","w")
# for cell in allcellsl10: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile10.write(cell2+"\n")

# Cellfile11 = open(Matlabfile + "/" + str(l10Micro) + "_" + str(l11Micro) + "_Cells.txt","w")
# for cell in allcellsl11: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile11.write(cell2+"\n")

# Cellfile12 = open(Matlabfile + "/" + str(l11Micro) + "_" + str(l12Micro) + "_Cells.txt","w")
# for cell in allcellsl12: 
# 	cell2 = str(cell)[1:-1]
# 	#print cell
# 	#print cell2
# 	Cellfile12.write(cell2+"\n")

icountl1 = len(aMikro_averagelistl1)
print "len amikro", len(aMikro_averagelistl1), len(bMikro_averagelistl1), len(lengthMikro_averagelistl1)
icountl2 = len(aMikro_averagelistl2)
icountl3 = len(aMikro_averagelistl3)
icountl4 = len(aMikro_averagelistl4)
icountl5 = len(aMikro_averagelistl5)
icountl6 = len(aMikro_averagelistl6)
icountl7 = len(aMikro_averagelistl7)
icountl8 = len(aMikro_averagelistl8)
icountl9 = len(aMikro_averagelistl9)
icountl10 = len(aMikro_averagelistl10)
icountl11 = len(aMikro_averagelistl11)
icountl12 = len(aMikro_averagelistl12)


aMikro_averagel1 = np.mean(aMikro_averagelistl1)
a2Mikro_averagel1 = np.mean(a2Mikro_averagelistl1)
bMikro_averagel1 = np.mean(bMikro_averagelistl1)
b2Mikro_averagel1 = np.mean(b2Mikro_averagelistl1)
areaMikro_averagel1 = np.mean(areaMikro_averagelistl1)
perimeterMikro_averagel1 = np.mean(perimeterMikro_averagelistl1)
lengthMikro_averagel1 = np.mean(lengthMikro_averagelistl1)

aMikro_variancel1 = np.var(aMikro_averagelistl1)
a2Mikro_variancel1 = np.var(a2Mikro_averagelistl1)
bMikro_variancel1 = np.var(bMikro_averagelistl1)
b2Mikro_variancel1 = np.var(b2Mikro_averagelistl1)
areaMikro_variancel1 = np.var(areaMikro_averagelistl1)
perimeterMikro_variancel1 = np.var(perimeterMikro_averagelistl1)
lengthMikro_variancel1 = np.var(lengthMikro_averagelistl1)

aMikro_sigmal1 = np.sqrt(aMikro_variancel1)
a2Mikro_sigmal1 = np.sqrt(a2Mikro_variancel1)
bMikro_sigmal1 = np.sqrt(bMikro_variancel1)
b2Mikro_sigmal1 = np.sqrt(b2Mikro_variancel1)
areaMikro_sigmal1 = np.sqrt(areaMikro_variancel1)
perimeterMikro_sigmal1 = np.sqrt(perimeterMikro_variancel1)
lengthMikro_sigmal1 = np.sqrt(lengthMikro_variancel1)

aMikro_standardfehlerl1 = aMikro_sigmal1/np.sqrt(icountl1)
a2Mikro_standardfehlerl1 = a2Mikro_sigmal1/np.sqrt(icountl1)
bMikro_standardfehlerl1 = bMikro_sigmal1/np.sqrt(icountl1)
b2Mikro_standardfehlerl1 = b2Mikro_sigmal1/np.sqrt(icountl1)
areaMikro_standardfehlerl1 = areaMikro_sigmal1/np.sqrt(icountl1)
perimeterMikro_standardfehlerl1 = perimeterMikro_sigmal1/np.sqrt(icountl1)
lengthMikro_standardfehlerl1 = lengthMikro_sigmal1/np.sqrt(icountl1)


#######l2

aMikro_averagel2 = np.mean(aMikro_averagelistl2)
a2Mikro_averagel2 = np.mean(a2Mikro_averagelistl2)
bMikro_averagel2 = np.mean(bMikro_averagelistl2)
b2Mikro_averagel2 = np.mean(b2Mikro_averagelistl2)
areaMikro_averagel2 = np.mean(areaMikro_averagelistl2)
perimeterMikro_averagel2 = np.mean(perimeterMikro_averagelistl2)
lengthMikro_averagel2 = np.mean(lengthMikro_averagelistl2)

aMikro_variancel2 = np.var(aMikro_averagelistl2)
a2Mikro_variancel2 = np.var(a2Mikro_averagelistl2)
bMikro_variancel2 = np.var(bMikro_averagelistl2)
b2Mikro_variancel2 = np.var(b2Mikro_averagelistl2)
areaMikro_variancel2 = np.var(areaMikro_averagelistl2)
perimeterMikro_variancel2 = np.var(perimeterMikro_averagelistl2)
lengthMikro_variancel2 = np.var(lengthMikro_averagelistl2)

aMikro_sigmal2 = np.sqrt(aMikro_variancel2)
a2Mikro_sigmal2 = np.sqrt(a2Mikro_variancel2)
bMikro_sigmal2 = np.sqrt(bMikro_variancel2)
b2Mikro_sigmal2 = np.sqrt(b2Mikro_variancel2)
areaMikro_sigmal2 = np.sqrt(areaMikro_variancel2)
perimeterMikro_sigmal2 = np.sqrt(perimeterMikro_variancel2)
lengthMikro_sigmal2 = np.sqrt(lengthMikro_variancel2)

aMikro_standardfehlerl2 = aMikro_sigmal2/np.sqrt(icountl2)
a2Mikro_standardfehlerl2 = a2Mikro_sigmal2/np.sqrt(icountl2)
bMikro_standardfehlerl2 = bMikro_sigmal2/np.sqrt(icountl2)
b2Mikro_standardfehlerl2 = b2Mikro_sigmal2/np.sqrt(icountl2)
areaMikro_standardfehlerl2 = areaMikro_sigmal2/np.sqrt(icountl2)
perimeterMikro_standardfehlerl2 = perimeterMikro_sigmal2/np.sqrt(icountl2)
lengthMikro_standardfehlerl2 = lengthMikro_sigmal2/np.sqrt(icountl2)

#########l3

aMikro_averagel3 = np.mean(aMikro_averagelistl3)
a2Mikro_averagel3 = np.mean(a2Mikro_averagelistl3)
bMikro_averagel3 = np.mean(bMikro_averagelistl3)
b2Mikro_averagel3 = np.mean(b2Mikro_averagelistl3)
areaMikro_averagel3 = np.mean(areaMikro_averagelistl3)
perimeterMikro_averagel3 = np.mean(perimeterMikro_averagelistl3)
lengthMikro_averagel3 = np.mean(lengthMikro_averagelistl3)

aMikro_variancel3 = np.var(aMikro_averagelistl3)
a2Mikro_variancel3 = np.var(a2Mikro_averagelistl3)
bMikro_variancel3 = np.var(bMikro_averagelistl3)
b2Mikro_variancel3 = np.var(b2Mikro_averagelistl3)
areaMikro_variancel3 = np.var(areaMikro_averagelistl3)
perimeterMikro_variancel3 = np.var(perimeterMikro_averagelistl3)
lengthMikro_variancel3 = np.var(lengthMikro_averagelistl3)

aMikro_sigmal3 = np.sqrt(aMikro_variancel3)
a2Mikro_sigmal3 = np.sqrt(a2Mikro_variancel3)
bMikro_sigmal3 = np.sqrt(bMikro_variancel3)
b2Mikro_sigmal3 = np.sqrt(b2Mikro_variancel3)
areaMikro_sigmal3 = np.sqrt(areaMikro_variancel3)
perimeterMikro_sigmal3 = np.sqrt(perimeterMikro_variancel3)
lengthMikro_sigmal3 = np.sqrt(lengthMikro_variancel3)

aMikro_standardfehlerl3 = aMikro_sigmal3/np.sqrt(icountl3)
a2Mikro_standardfehlerl3 = a2Mikro_sigmal3/np.sqrt(icountl3)
bMikro_standardfehlerl3 = bMikro_sigmal3/np.sqrt(icountl3)
b2Mikro_standardfehlerl3 = b2Mikro_sigmal3/np.sqrt(icountl3)
areaMikro_standardfehlerl3 = areaMikro_sigmal3/np.sqrt(icountl3)
perimeterMikro_standardfehlerl3 = perimeterMikro_sigmal3/np.sqrt(icountl3)
lengthMikro_standardfehlerl3 = lengthMikro_sigmal3/np.sqrt(icountl3)

#####l4

aMikro_averagel4 = np.mean(aMikro_averagelistl4)
a2Mikro_averagel4 = np.mean(a2Mikro_averagelistl4)
bMikro_averagel4 = np.mean(bMikro_averagelistl4)
b2Mikro_averagel4 = np.mean(b2Mikro_averagelistl4)
areaMikro_averagel4 = np.mean(areaMikro_averagelistl4)
perimeterMikro_averagel4 = np.mean(perimeterMikro_averagelistl4)
lengthMikro_averagel4 = np.mean(lengthMikro_averagelistl4)

aMikro_variancel4 = np.var(aMikro_averagelistl4)
a2Mikro_variancel4 = np.var(a2Mikro_averagelistl4)
bMikro_variancel4 = np.var(bMikro_averagelistl4)
b2Mikro_variancel4 = np.var(b2Mikro_averagelistl4)
areaMikro_variancel4 = np.var(areaMikro_averagelistl4)
perimeterMikro_variancel4 = np.var(perimeterMikro_averagelistl4)
lengthMikro_variancel4 = np.var(lengthMikro_averagelistl4)

aMikro_sigmal4 = np.sqrt(aMikro_variancel4)
a2Mikro_sigmal4 = np.sqrt(a2Mikro_variancel4)
bMikro_sigmal4 = np.sqrt(bMikro_variancel4)
b2Mikro_sigmal4 = np.sqrt(b2Mikro_variancel4)
areaMikro_sigmal4 = np.sqrt(areaMikro_variancel4)
perimeterMikro_sigmal4 = np.sqrt(perimeterMikro_variancel4)
lengthMikro_sigmal4 = np.sqrt(lengthMikro_variancel4)

aMikro_standardfehlerl4 = aMikro_sigmal4/np.sqrt(icountl4)
a2Mikro_standardfehlerl4 = a2Mikro_sigmal4/np.sqrt(icountl4)
bMikro_standardfehlerl4 = bMikro_sigmal4/np.sqrt(icountl4)
b2Mikro_standardfehlerl4 = b2Mikro_sigmal4/np.sqrt(icountl4)
areaMikro_standardfehlerl4 = areaMikro_sigmal4/np.sqrt(icountl4)
perimeterMikro_standardfehlerl4 = perimeterMikro_sigmal4/np.sqrt(icountl4)
lengthMikro_standardfehlerl4 = lengthMikro_sigmal4/np.sqrt(icountl4)


########l5

aMikro_averagel5 = np.mean(aMikro_averagelistl5)
a2Mikro_averagel5 = np.mean(a2Mikro_averagelistl5)
bMikro_averagel5 = np.mean(bMikro_averagelistl5)
b2Mikro_averagel5 = np.mean(b2Mikro_averagelistl5)
areaMikro_averagel5 = np.mean(areaMikro_averagelistl5)
perimeterMikro_averagel5 = np.mean(perimeterMikro_averagelistl5)
lengthMikro_averagel5 = np.mean(lengthMikro_averagelistl5)

aMikro_variancel5 = np.var(aMikro_averagelistl5)
a2Mikro_variancel5 = np.var(a2Mikro_averagelistl5)
bMikro_variancel5 = np.var(bMikro_averagelistl5)
b2Mikro_variancel5 = np.var(b2Mikro_averagelistl5)
areaMikro_variancel5 = np.var(areaMikro_averagelistl5)
perimeterMikro_variancel5 = np.var(perimeterMikro_averagelistl5)
lengthMikro_variancel5 = np.var(lengthMikro_averagelistl5)

aMikro_sigmal5 = np.sqrt(aMikro_variancel5)
a2Mikro_sigmal5 = np.sqrt(a2Mikro_variancel5)
bMikro_sigmal5 = np.sqrt(bMikro_variancel5)
b2Mikro_sigmal5 = np.sqrt(b2Mikro_variancel5)
areaMikro_sigmal5 = np.sqrt(areaMikro_variancel5)
perimeterMikro_sigmal5 = np.sqrt(perimeterMikro_variancel5)
lengthMikro_sigmal5 = np.sqrt(lengthMikro_variancel5)

aMikro_standardfehlerl5 = aMikro_sigmal5/np.sqrt(icountl5)
a2Mikro_standardfehlerl5 = a2Mikro_sigmal5/np.sqrt(icountl5)
bMikro_standardfehlerl5 = bMikro_sigmal5/np.sqrt(icountl5)
b2Mikro_standardfehlerl5 = b2Mikro_sigmal5/np.sqrt(icountl5)
areaMikro_standardfehlerl5 = areaMikro_sigmal5/np.sqrt(icountl5)
perimeterMikro_standardfehlerl5 = perimeterMikro_sigmal5/np.sqrt(icountl5)
lengthMikro_standardfehlerl5 = lengthMikro_sigmal5/np.sqrt(icountl5)


#######l6

aMikro_averagel6 = np.mean(aMikro_averagelistl6)
a2Mikro_averagel6 = np.mean(a2Mikro_averagelistl6)
bMikro_averagel6 = np.mean(bMikro_averagelistl6)
b2Mikro_averagel6 = np.mean(b2Mikro_averagelistl6)
areaMikro_averagel6 = np.mean(areaMikro_averagelistl6)
perimeterMikro_averagel6 = np.mean(perimeterMikro_averagelistl6)
lengthMikro_averagel6 = np.mean(lengthMikro_averagelistl6)

aMikro_variancel6 = np.var(aMikro_averagelistl6)
a2Mikro_variancel6 = np.var(a2Mikro_averagelistl6)
bMikro_variancel6 = np.var(bMikro_averagelistl6)
b2Mikro_variancel6 = np.var(b2Mikro_averagelistl6)
areaMikro_variancel6 = np.var(areaMikro_averagelistl6)
perimeterMikro_variancel6 = np.var(perimeterMikro_averagelistl6)
lengthMikro_variancel6 = np.var(lengthMikro_averagelistl6)

aMikro_sigmal6 = np.sqrt(aMikro_variancel6)
a2Mikro_sigmal6 = np.sqrt(a2Mikro_variancel6)
bMikro_sigmal6 = np.sqrt(bMikro_variancel6)
b2Mikro_sigmal6 = np.sqrt(b2Mikro_variancel6)
areaMikro_sigmal6 = np.sqrt(areaMikro_variancel6)
perimeterMikro_sigmal6 = np.sqrt(perimeterMikro_variancel6)
lengthMikro_sigmal6 = np.sqrt(lengthMikro_variancel6)

aMikro_standardfehlerl6 = aMikro_sigmal6/np.sqrt(icountl6)
a2Mikro_standardfehlerl6 = a2Mikro_sigmal6/np.sqrt(icountl6)
bMikro_standardfehlerl6 = bMikro_sigmal6/np.sqrt(icountl6)
b2Mikro_standardfehlerl6 = b2Mikro_sigmal6/np.sqrt(icountl6)
areaMikro_standardfehlerl6 = areaMikro_sigmal6/np.sqrt(icountl6)
perimeterMikro_standardfehlerl6 = perimeterMikro_sigmal6/np.sqrt(icountl6)
lengthMikro_standardfehlerl6 = lengthMikro_sigmal6/np.sqrt(icountl6)


######l7

aMikro_averagel7 = np.mean(aMikro_averagelistl7)
a2Mikro_averagel7 = np.mean(a2Mikro_averagelistl7)
bMikro_averagel7 = np.mean(bMikro_averagelistl7)
b2Mikro_averagel7 = np.mean(b2Mikro_averagelistl7)
areaMikro_averagel7 = np.mean(areaMikro_averagelistl7)
perimeterMikro_averagel7 = np.mean(perimeterMikro_averagelistl7)
lengthMikro_averagel7 = np.mean(lengthMikro_averagelistl7)

aMikro_variancel7 = np.var(aMikro_averagelistl7)
a2Mikro_variancel7 = np.var(a2Mikro_averagelistl7)
bMikro_variancel7 = np.var(bMikro_averagelistl7)
b2Mikro_variancel7 = np.var(b2Mikro_averagelistl7)
areaMikro_variancel7 = np.var(areaMikro_averagelistl7)
perimeterMikro_variancel7 = np.var(perimeterMikro_averagelistl7)
lengthMikro_variancel7 = np.var(lengthMikro_averagelistl7)

aMikro_sigmal7 = np.sqrt(aMikro_variancel7)
a2Mikro_sigmal7 = np.sqrt(a2Mikro_variancel7)
bMikro_sigmal7 = np.sqrt(bMikro_variancel7)
b2Mikro_sigmal7 = np.sqrt(b2Mikro_variancel7)
areaMikro_sigmal7 = np.sqrt(areaMikro_variancel7)
perimeterMikro_sigmal7 = np.sqrt(perimeterMikro_variancel7)
lengthMikro_sigmal7 = np.sqrt(lengthMikro_variancel7)

aMikro_standardfehlerl7 = aMikro_sigmal7/np.sqrt(icountl7)
a2Mikro_standardfehlerl7 = a2Mikro_sigmal7/np.sqrt(icountl7)
bMikro_standardfehlerl7 = bMikro_sigmal7/np.sqrt(icountl7)
b2Mikro_standardfehlerl7 = b2Mikro_sigmal7/np.sqrt(icountl7)
areaMikro_standardfehlerl7 = areaMikro_sigmal7/np.sqrt(icountl7)
perimeterMikro_standardfehlerl7 = perimeterMikro_sigmal7/np.sqrt(icountl7)
lengthMikro_standardfehlerl7 = lengthMikro_sigmal7/np.sqrt(icountl7)


######l8

aMikro_averagel8 = np.mean(aMikro_averagelistl8)
a2Mikro_averagel8 = np.mean(a2Mikro_averagelistl8)
bMikro_averagel8 = np.mean(bMikro_averagelistl8)
b2Mikro_averagel8 = np.mean(b2Mikro_averagelistl8)
areaMikro_averagel8 = np.mean(areaMikro_averagelistl8)
perimeterMikro_averagel8 = np.mean(perimeterMikro_averagelistl8)
lengthMikro_averagel8 = np.mean(lengthMikro_averagelistl8)

aMikro_variancel8 = np.var(aMikro_averagelistl8)
a2Mikro_variancel8 = np.var(a2Mikro_averagelistl8)
bMikro_variancel8 = np.var(bMikro_averagelistl8)
b2Mikro_variancel8 = np.var(b2Mikro_averagelistl8)
areaMikro_variancel8 = np.var(areaMikro_averagelistl8)
perimeterMikro_variancel8 = np.var(perimeterMikro_averagelistl8)
lengthMikro_variancel8 = np.var(lengthMikro_averagelistl8)

aMikro_sigmal8 = np.sqrt(aMikro_variancel8)
a2Mikro_sigmal8 = np.sqrt(a2Mikro_variancel8)
bMikro_sigmal8 = np.sqrt(bMikro_variancel8)
b2Mikro_sigmal8 = np.sqrt(b2Mikro_variancel8)
areaMikro_sigmal8 = np.sqrt(areaMikro_variancel8)
perimeterMikro_sigmal8 = np.sqrt(perimeterMikro_variancel8)
lengthMikro_sigmal8 = np.sqrt(lengthMikro_variancel8)

aMikro_standardfehlerl8 = aMikro_sigmal8/np.sqrt(icountl8)
a2Mikro_standardfehlerl8 = a2Mikro_sigmal8/np.sqrt(icountl8)
bMikro_standardfehlerl8 = bMikro_sigmal8/np.sqrt(icountl8)
b2Mikro_standardfehlerl8 = b2Mikro_sigmal8/np.sqrt(icountl8)
areaMikro_standardfehlerl8 = areaMikro_sigmal8/np.sqrt(icountl8)
perimeterMikro_standardfehlerl8 = perimeterMikro_sigmal8/np.sqrt(icountl8)
lengthMikro_standardfehlerl8 = lengthMikro_sigmal8/np.sqrt(icountl8)


####l9

aMikro_averagel9 = np.mean(aMikro_averagelistl9)
a2Mikro_averagel9 = np.mean(a2Mikro_averagelistl9)
bMikro_averagel9 = np.mean(bMikro_averagelistl9)
b2Mikro_averagel9 = np.mean(b2Mikro_averagelistl9)
areaMikro_averagel9 = np.mean(areaMikro_averagelistl9)
perimeterMikro_averagel9 = np.mean(perimeterMikro_averagelistl9)
lengthMikro_averagel9 = np.mean(lengthMikro_averagelistl9)

aMikro_variancel9 = np.var(aMikro_averagelistl9)
a2Mikro_variancel9 = np.var(a2Mikro_averagelistl9)
bMikro_variancel9 = np.var(bMikro_averagelistl9)
b2Mikro_variancel9 = np.var(b2Mikro_averagelistl9)
areaMikro_variancel9 = np.var(areaMikro_averagelistl9)
perimeterMikro_variancel9 = np.var(perimeterMikro_averagelistl9)
lengthMikro_variancel9 = np.var(lengthMikro_averagelistl9)

aMikro_sigmal9 = np.sqrt(aMikro_variancel9)
a2Mikro_sigmal9 = np.sqrt(a2Mikro_variancel9)
bMikro_sigmal9 = np.sqrt(bMikro_variancel9)
b2Mikro_sigmal9 = np.sqrt(b2Mikro_variancel9)
areaMikro_sigmal9 = np.sqrt(areaMikro_variancel9)
perimeterMikro_sigmal9 = np.sqrt(perimeterMikro_variancel9)
lengthMikro_sigmal9 = np.sqrt(lengthMikro_variancel9)

aMikro_standardfehlerl9 = aMikro_sigmal9/np.sqrt(icountl9)
a2Mikro_standardfehlerl9 = a2Mikro_sigmal9/np.sqrt(icountl9)
bMikro_standardfehlerl9 = bMikro_sigmal9/np.sqrt(icountl9)
b2Mikro_standardfehlerl9 = b2Mikro_sigmal9/np.sqrt(icountl9)
areaMikro_standardfehlerl9 = areaMikro_sigmal9/np.sqrt(icountl9)
perimeterMikro_standardfehlerl9 = perimeterMikro_sigmal9/np.sqrt(icountl9)
lengthMikro_standardfehlerl9 = lengthMikro_sigmal9/np.sqrt(icountl9)


#####l10

aMikro_averagel10 = np.mean(aMikro_averagelistl10)
a2Mikro_averagel10 = np.mean(a2Mikro_averagelistl10)
bMikro_averagel10 = np.mean(bMikro_averagelistl10)
b2Mikro_averagel10 = np.mean(b2Mikro_averagelistl10)
areaMikro_averagel10 = np.mean(areaMikro_averagelistl10)
perimeterMikro_averagel10 = np.mean(perimeterMikro_averagelistl10)
lengthMikro_averagel10 = np.mean(lengthMikro_averagelistl10)

aMikro_variancel10 = np.var(aMikro_averagelistl10)
a2Mikro_variancel10 = np.var(a2Mikro_averagelistl10)
bMikro_variancel10 = np.var(bMikro_averagelistl10)
b2Mikro_variancel10 = np.var(b2Mikro_averagelistl10)
areaMikro_variancel10 = np.var(areaMikro_averagelistl10)
perimeterMikro_variancel10 = np.var(perimeterMikro_averagelistl10)
lengthMikro_variancel10 = np.var(lengthMikro_averagelistl10)

aMikro_sigmal10 = np.sqrt(aMikro_variancel10)
a2Mikro_sigmal10 = np.sqrt(a2Mikro_variancel10)
bMikro_sigmal10 = np.sqrt(bMikro_variancel10)
b2Mikro_sigmal10 = np.sqrt(b2Mikro_variancel10)
areaMikro_sigmal10 = np.sqrt(areaMikro_variancel10)
perimeterMikro_sigmal10 = np.sqrt(perimeterMikro_variancel10)
lengthMikro_sigmal10 = np.sqrt(lengthMikro_variancel10)

aMikro_standardfehlerl10 = aMikro_sigmal10/np.sqrt(icountl10)
a2Mikro_standardfehlerl10 = a2Mikro_sigmal10/np.sqrt(icountl10)
bMikro_standardfehlerl10 = bMikro_sigmal10/np.sqrt(icountl10)
b2Mikro_standardfehlerl10 = b2Mikro_sigmal10/np.sqrt(icountl10)
areaMikro_standardfehlerl10 = areaMikro_sigmal10/np.sqrt(icountl10)
perimeterMikro_standardfehlerl10 = perimeterMikro_sigmal10/np.sqrt(icountl10)
lengthMikro_standardfehlerl10 = lengthMikro_sigmal10/np.sqrt(icountl10)


#######l11

aMikro_averagel11 = np.mean(aMikro_averagelistl11)
a2Mikro_averagel11 = np.mean(a2Mikro_averagelistl11)
bMikro_averagel11 = np.mean(bMikro_averagelistl11)
b2Mikro_averagel11 = np.mean(b2Mikro_averagelistl11)
areaMikro_averagel11 = np.mean(areaMikro_averagelistl11)
perimeterMikro_averagel11 = np.mean(perimeterMikro_averagelistl11)
lengthMikro_averagel11 = np.mean(lengthMikro_averagelistl11)

aMikro_variancel11 = np.var(aMikro_averagelistl11)
a2Mikro_variancel11 = np.var(a2Mikro_averagelistl11)
bMikro_variancel11 = np.var(bMikro_averagelistl11)
b2Mikro_variancel11 = np.var(b2Mikro_averagelistl11)
areaMikro_variancel11 = np.var(areaMikro_averagelistl11)
perimeterMikro_variancel11 = np.var(perimeterMikro_averagelistl11)
lengthMikro_variancel11 = np.var(lengthMikro_averagelistl11)

aMikro_sigmal11 = np.sqrt(aMikro_variancel11)
a2Mikro_sigmal11 = np.sqrt(a2Mikro_variancel11)
bMikro_sigmal11 = np.sqrt(bMikro_variancel11)
b2Mikro_sigmal11 = np.sqrt(b2Mikro_variancel11)
areaMikro_sigmal11 = np.sqrt(areaMikro_variancel11)
perimeterMikro_sigmal11 = np.sqrt(perimeterMikro_variancel11)
lengthMikro_sigmal11 = np.sqrt(lengthMikro_variancel11)

aMikro_standardfehlerl11 = aMikro_sigmal11/np.sqrt(icountl11)
a2Mikro_standardfehlerl11 = a2Mikro_sigmal11/np.sqrt(icountl11)
bMikro_standardfehlerl11 = bMikro_sigmal11/np.sqrt(icountl11)
b2Mikro_standardfehlerl11 = b2Mikro_sigmal11/np.sqrt(icountl11)
areaMikro_standardfehlerl11 = areaMikro_sigmal11/np.sqrt(icountl11)
perimeterMikro_standardfehlerl11 = perimeterMikro_sigmal11/np.sqrt(icountl11)
lengthMikro_standardfehlerl11 = lengthMikro_sigmal11/np.sqrt(icountl11)


#####l12

aMikro_averagel12 = np.mean(aMikro_averagelistl12)
a2Mikro_averagel12 = np.mean(a2Mikro_averagelistl12)
bMikro_averagel12 = np.mean(bMikro_averagelistl12)
b2Mikro_averagel12 = np.mean(b2Mikro_averagelistl12)
areaMikro_averagel12 = np.mean(areaMikro_averagelistl12)
perimeterMikro_averagel12 = np.mean(perimeterMikro_averagelistl12)
lengthMikro_averagel12 = np.mean(lengthMikro_averagelistl12)

aMikro_variancel12 = np.var(aMikro_averagelistl12)
a2Mikro_variancel12 = np.var(a2Mikro_averagelistl12)
bMikro_variancel12 = np.var(bMikro_averagelistl12)
b2Mikro_variancel12 = np.var(b2Mikro_averagelistl12)
areaMikro_variancel12 = np.var(areaMikro_averagelistl12)
perimeterMikro_variancel12 = np.var(perimeterMikro_averagelistl12)
lengthMikro_variancel12 = np.var(lengthMikro_averagelistl12)

aMikro_sigmal12 = np.sqrt(aMikro_variancel12)
a2Mikro_sigmal12 = np.sqrt(a2Mikro_variancel12)
bMikro_sigmal12 = np.sqrt(bMikro_variancel12)
b2Mikro_sigmal12 = np.sqrt(b2Mikro_variancel12)
areaMikro_sigmal12 = np.sqrt(areaMikro_variancel12)
perimeterMikro_sigmal12 = np.sqrt(perimeterMikro_variancel12)
lengthMikro_sigmal12 = np.sqrt(lengthMikro_variancel12)

aMikro_standardfehlerl12 = aMikro_sigmal12/np.sqrt(icountl12)
a2Mikro_standardfehlerl12 = a2Mikro_sigmal12/np.sqrt(icountl12)
bMikro_standardfehlerl12 = bMikro_sigmal12/np.sqrt(icountl12)
b2Mikro_standardfehlerl12 = b2Mikro_sigmal12/np.sqrt(icountl12)
areaMikro_standardfehlerl12 = areaMikro_sigmal12/np.sqrt(icountl12)
perimeterMikro_standardfehlerl12 = perimeterMikro_sigmal12/np.sqrt(icountl12)
lengthMikro_standardfehlerl12 = lengthMikro_sigmal12/np.sqrt(icountl12)





wf.write("Parameter: " + "Value" + "Variance" + "Sigma" + "Standard error"+ "\n")
wf.write(str(minlengthMicro)+ " to " + str(l1Micro)+"\n")
wf.write("Area in micrometer: " + str(areaMikro_averagel1.round(2)) +", "+ str(areaMikro_variancel1.round(2))+ ", "+str(areaMikro_sigmal1.round(2))+ ", "+ str(areaMikro_standardfehlerl1.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel1.round(2)) +", "+str(perimeterMikro_variancel1.round(2))+ ", "+str(perimeterMikro_sigmal1.round(2))+ ", "+ str(perimeterMikro_standardfehlerl1.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel1.round(2)) +", "+ str(lengthMikro_variancel1.round(2))+ ", "+str(lengthMikro_sigmal1.round(2))+ ", "+ str(lengthMikro_standardfehlerl1.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel1.round(2)) +", "+ str(aMikro_variancel1.round(2))+ ", "+str(aMikro_sigmal1.round(2))+ ", "+ str(aMikro_standardfehlerl1.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel1.round(2))+", " +str(bMikro_variancel1.round(2))+ ", "+str(bMikro_sigmal1.round(2))+ ", "+ str(bMikro_standardfehlerl1.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel1.round(2)) +", "+ str(a2Mikro_variancel1.round(2))+ ", "+str(a2Mikro_sigmal1.round(2))+ ", "+ str(a2Mikro_standardfehlerl1.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel1.round(2))+", " + str(b2Mikro_variancel1.round(2))+ ", "+str(b2Mikro_sigmal1.round(2))+ ", "+ str(b2Mikro_standardfehlerl1.round(2))+"\n")
wf.write("Amount: " + str(icountl1) + "\n"+ "\n")


wf.write(str(l1Micro)+ " to " + str(l2Micro)+"\n")

wf.write("Area in micrometer: " + str(areaMikro_averagel2.round(2)) +", "+ str(areaMikro_variancel2.round(2))+ ", "+str(areaMikro_sigmal2.round(2))+ ", "+ str(areaMikro_standardfehlerl2.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel2.round(2))+", " +str(perimeterMikro_variancel2.round(2))+ ", "+str(perimeterMikro_sigmal2.round(2))+ ", "+ str(perimeterMikro_standardfehlerl2.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel2.round(2)) +", "+ str(lengthMikro_variancel2.round(2))+ ", "+str(lengthMikro_sigmal2.round(2))+ ", "+ str(lengthMikro_standardfehlerl2.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel2.round(2))+", " + str(aMikro_variancel2.round(2))+ ", "+str(aMikro_sigmal2.round(2))+ ", "+ str(aMikro_standardfehlerl2.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel2.round(2))+", " +str(bMikro_variancel2.round(2))+ ", "+str(bMikro_sigmal2.round(2))+ ", "+ str(bMikro_standardfehlerl2.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel2.round(2))+", " + str(a2Mikro_variancel2.round(2))+ ", "+str(a2Mikro_sigmal2.round(2))+ ", "+ str(a2Mikro_standardfehlerl2.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel2.round(2)) +", "+", "+ str(b2Mikro_variancel2.round(2))+ ", "+str(b2Mikro_sigmal2.round(2))+ ", "+ str(b2Mikro_standardfehlerl2.round(2))+"\n" )
wf.write("Amount: " + str(icountl2) + "\n"+ "\n")


wf.write(str(l2Micro)+ " to " + str(l3Micro)+"\n")

wf.write("Area in micrometer: " + str(areaMikro_averagel3.round(2))+", " + str(areaMikro_variancel3.round(2))+ ", "+str(areaMikro_sigmal3.round(2))+ ", "+ str(areaMikro_standardfehlerl3.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel3.round(2))+", " +str(perimeterMikro_variancel3.round(2))+ ", "+str(perimeterMikro_sigmal3.round(2))+ ", "+ str(perimeterMikro_standardfehlerl3.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel3.round(2))+", " + str(lengthMikro_variancel3.round(2))+ ", "+str(lengthMikro_sigmal3.round(2))+ ", "+ str(lengthMikro_standardfehlerl3.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel3.round(2))+", " + str(aMikro_variancel3.round(2))+ ", "+str(aMikro_sigmal3.round(2))+ ", "+ str(aMikro_standardfehlerl3.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel3.round(2))+", " +str(bMikro_variancel3.round(2))+ ", "+str(bMikro_sigmal3.round(2))+ ", "+ str(bMikro_standardfehlerl3.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel3.round(2))+", " + str(a2Mikro_variancel3.round(2))+ ", "+str(a2Mikro_sigmal3.round(2))+ ", "+ str(a2Mikro_standardfehlerl3.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel3.round(2)) +", "+ str(b2Mikro_variancel3.round(2))+ ", "+str(b2Mikro_sigmal3.round(2))+ ", "+ str(b2Mikro_standardfehlerl3.round(2))+"\n")
wf.write("Amount: " + str(icountl3) + "\n"+ "\n")


wf.write(str(l3Micro)+ " to " + str(l4Micro)+"\n")

wf.write("Area in micrometer: " + str(areaMikro_averagel4.round(2)) +", "+ str(areaMikro_variancel4.round(2))+ ", "+str(areaMikro_sigmal4.round(2))+ ", "+ str(areaMikro_standardfehlerl4.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel4.round(2))+", " +str(perimeterMikro_variancel4.round(2))+ ", "+str(perimeterMikro_sigmal4.round(2))+ ", "+ str(perimeterMikro_standardfehlerl4.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel4.round(2))+", " + str(lengthMikro_variancel4.round(2))+ ", "+str(lengthMikro_sigmal4.round(2))+ ", "+ str(lengthMikro_standardfehlerl4.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel4.round(2))+", " + str(aMikro_variancel4.round(2))+ ", "+str(aMikro_sigmal4.round(2))+ ", "+ str(aMikro_standardfehlerl4.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel4.round(2)) +", "+str(bMikro_variancel4.round(2))+ ", "+str(bMikro_sigmal4.round(2))+ ", "+ str(bMikro_standardfehlerl4.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel4.round(2))+", " + str(a2Mikro_variancel4.round(2))+ ", "+str(a2Mikro_sigmal4.round(2))+ ", "+ str(a2Mikro_standardfehlerl4.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel4.round(2)) +", "+ str(b2Mikro_variancel4.round(2))+ ", "+str(b2Mikro_sigmal4.round(2))+ ", "+ str(b2Mikro_standardfehlerl4.round(2))+"\n")
wf.write("Amount: " + str(icountl4) + "\n"+ "\n")


wf.write(str(l4Micro)+ " to " + str(l5Micro)+"\n")


wf.write("Area in micrometer: " + str(areaMikro_averagel5.round(2)) +", "+ str(areaMikro_variancel5.round(2))+ ", "+str(areaMikro_sigmal5.round(2))+ ", "+ str(areaMikro_standardfehlerl5.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel5.round(2))+", " +str(perimeterMikro_variancel5.round(2))+ ", "+str(perimeterMikro_sigmal5.round(2))+ ", "+ str(perimeterMikro_standardfehlerl5.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel5.round(2))+", " + str(lengthMikro_variancel5.round(2))+ ", "+str(lengthMikro_sigmal5.round(2))+ ", "+ str(lengthMikro_standardfehlerl5.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel5.round(2)) +", "+ str(aMikro_variancel5.round(2))+ ", "+str(aMikro_sigmal5.round(2))+ ", "+ str(aMikro_standardfehlerl5.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel5.round(2))+", " +str(bMikro_variancel5.round(2))+ ", "+str(bMikro_sigmal5.round(2))+ ", "+ str(bMikro_standardfehlerl5.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel5.round(2)) +", "+ str(a2Mikro_variancel5.round(2))+ ", "+str(a2Mikro_sigmal5.round(2))+ ", "+ str(a2Mikro_standardfehlerl5.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel5.round(2)) +", "+ str(b2Mikro_variancel5.round(2))+ ", "+str(b2Mikro_sigmal5.round(2))+ ", "+ str(b2Mikro_standardfehlerl5.round(2))+"\n")
wf.write("Amount: " + str(icountl5) + "\n"+ "\n")


wf.write(str(l5Micro)+ " to " + str(l6Micro)+"\n")

wf.write("Area in micrometer: " + str(areaMikro_averagel6.round(2))+", " + str(areaMikro_variancel6.round(2))+ ", "+str(areaMikro_sigmal6.round(2))+ ", "+ str(areaMikro_standardfehlerl6.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel6.round(2))+", " +str(perimeterMikro_variancel6.round(2))+ ", "+str(perimeterMikro_sigmal6.round(2))+ ", "+ str(perimeterMikro_standardfehlerl6.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel6.round(2))+", " + str(lengthMikro_variancel6.round(2))+ ", "+str(lengthMikro_sigmal6.round(2))+ ", "+ str(lengthMikro_standardfehlerl6.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel6.round(2)) +", "+ str(aMikro_variancel6.round(2))+ ", "+str(aMikro_sigmal6.round(2))+ ", "+ str(aMikro_standardfehlerl6.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel6.round(2))+", " +str(bMikro_variancel6.round(2))+ ", "+str(bMikro_sigmal6.round(2))+ ", "+ str(bMikro_standardfehlerl6.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel6.round(2))+", " + str(a2Mikro_variancel6.round(2))+ ", "+str(a2Mikro_sigmal6.round(2))+ ", "+ str(a2Mikro_standardfehlerl6.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel6.round(2)) +", "+ str(b2Mikro_variancel6.round(2))+ ", "+str(b2Mikro_sigmal6.round(2))+ ", "+ str(b2Mikro_standardfehlerl6.round(2))+"\n")
wf.write("Amount: " + str(icountl6) + "\n"+ "\n")


wf.write(str(l6Micro)+ " to " + str(l7Micro)+"\n")


wf.write("Area in micrometer: " + str(areaMikro_averagel7.round(2))+", " + str(areaMikro_variancel7.round(2))+ ", "+str(areaMikro_sigmal7.round(2))+ ", "+ str(areaMikro_standardfehlerl7.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel7.round(2)) +", "+str(perimeterMikro_variancel7.round(2))+ ", "+str(perimeterMikro_sigmal7.round(2))+ ", "+ str(perimeterMikro_standardfehlerl7.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel7.round(2))+", " + str(lengthMikro_variancel7.round(2))+ ", "+str(lengthMikro_sigmal7.round(2))+ ", "+ str(lengthMikro_standardfehlerl7.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel7.round(2)) +", "+ str(aMikro_variancel7.round(2))+ ", "+str(aMikro_sigmal7.round(2))+ ", "+ str(aMikro_standardfehlerl7.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel7.round(2))+", " +str(bMikro_variancel7.round(2))+ ", "+str(bMikro_sigmal7.round(2))+ ", "+ str(bMikro_standardfehlerl7.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel7.round(2))+", " + str(a2Mikro_variancel7.round(2))+ ", "+str(a2Mikro_sigmal7.round(2))+ ", "+ str(a2Mikro_standardfehlerl7.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel7.round(2)) +", "+ str(b2Mikro_variancel7.round(2))+ ", "+str(b2Mikro_sigmal7.round(2))+ ", "+ str(b2Mikro_standardfehlerl7.round(2))+"\n")
wf.write("Amount: " + str(icountl7) + "\n"+ "\n")


wf.write(str(l7Micro)+ " to " + str(l8Micro)+"\n")

wf.write("Area in micrometer: " + str(areaMikro_averagel8.round(2)) +", "+ str(areaMikro_variancel8.round(2))+ ", "+str(areaMikro_sigmal8.round(2))+ ", "+ str(areaMikro_standardfehlerl8.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel8.round(2))+", " +str(perimeterMikro_variancel8.round(2))+ ", "+str(perimeterMikro_sigmal8.round(2))+ ", "+ str(perimeterMikro_standardfehlerl8.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel8.round(2)) +", "+ str(lengthMikro_variancel8.round(2))+ ", "+str(lengthMikro_sigmal8.round(2))+ ", "+ str(lengthMikro_standardfehlerl8.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel8.round(2))+", " + str(aMikro_variancel8.round(2))+ ", "+str(aMikro_sigmal8.round(2))+ ", "+ str(aMikro_standardfehlerl8.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel8.round(2)) +", "+str(bMikro_variancel8.round(2))+ ", "+str(bMikro_sigmal8.round(2))+ ", "+ str(bMikro_standardfehlerl8.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel8.round(2)) +", "+ str(a2Mikro_variancel8.round(2))+ ", "+str(a2Mikro_sigmal8.round(2))+ ", "+ str(a2Mikro_standardfehlerl8.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel8.round(2)) +", "+ str(b2Mikro_variancel8.round(2))+ ", "+str(b2Mikro_sigmal8.round(2))+ ", "+ str(b2Mikro_standardfehlerl8.round(2))+"\n")
wf.write("Amount: " + str(icountl8) + "\n"+ "\n")



wf.write(str(l8Micro)+ " to " + str(l9Micro)+"\n")

wf.write("Area in micrometer: " + str(areaMikro_averagel9.round(2))+", " + str(areaMikro_variancel9.round(2))+ ", "+str(areaMikro_sigmal9.round(2))+ ", "+ str(areaMikro_standardfehlerl9.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel9.round(2))+", " +str(perimeterMikro_variancel9.round(2))+ ", "+str(perimeterMikro_sigmal9.round(2))+ ", "+ str(perimeterMikro_standardfehlerl9.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel9.round(2)) +", "+ str(lengthMikro_variancel9.round(2))+ ", "+str(lengthMikro_sigmal9.round(2))+ ", "+ str(lengthMikro_standardfehlerl9.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel9.round(2))+", " + str(aMikro_variancel9.round(2))+ ", "+str(aMikro_sigmal9.round(2))+ ", "+ str(aMikro_standardfehlerl9.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel9.round(2))+", " +str(bMikro_variancel9.round(2))+ ", "+str(bMikro_sigmal9.round(2))+ ", "+ str(bMikro_standardfehlerl9.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel9.round(2))+", " + str(a2Mikro_variancel9.round(2))+ ", "+str(a2Mikro_sigmal9.round(2))+ ", "+ str(a2Mikro_standardfehlerl9.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel9.round(2))+", " + str(b2Mikro_variancel9.round(2))+ ", "+str(b2Mikro_sigmal9.round(2))+ ", "+ str(b2Mikro_standardfehlerl9.round(2))+"\n")
wf.write("Amount: " + str(icountl9) + "\n"+ "\n")



wf.write(str(l9Micro)+ " to " + str(l10Micro)+"\n")


wf.write("Area in micrometer: " + str(areaMikro_averagel10.round(2))+", " + str(areaMikro_variancel10.round(2))+ ", "+str(areaMikro_sigmal10.round(2))+ ", "+ str(areaMikro_standardfehlerl10.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel10.round(2))+", " +str(perimeterMikro_variancel10.round(2))+ ", "+str(perimeterMikro_sigmal10.round(2))+ ", "+ str(perimeterMikro_standardfehlerl10.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel10.round(2))+", " + str(lengthMikro_variancel10.round(2))+ ", "+str(lengthMikro_sigmal10.round(2))+ ", "+ str(lengthMikro_standardfehlerl10.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel10.round(2))+", " + str(aMikro_variancel10.round(2))+ ", "+str(aMikro_sigmal10.round(2))+ ", "+ str(aMikro_standardfehlerl10.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel10.round(2))+", " +str(bMikro_variancel10.round(2))+ ", "+str(bMikro_sigmal10.round(2))+ ", "+ str(bMikro_standardfehlerl10.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel10.round(2))+", "+ str(a2Mikro_variancel10.round(2))+ ", "+str(a2Mikro_sigmal10.round(2))+ ", "+ str(a2Mikro_standardfehlerl10.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel10.round(2)) +", "+ str(b2Mikro_variancel10.round(2))+ ", "+str(b2Mikro_sigmal10.round(2))+ ", "+ str(b2Mikro_standardfehlerl10.round(2))+"\n")
wf.write("Amount: " + str(icountl10) + "\n"+ "\n")


wf.write(str(l10Micro)+ " to " + str(l11Micro)+"\n")


wf.write("Area in micrometer: " + str(areaMikro_averagel11.round(2))+", " + str(areaMikro_variancel11.round(2))+ ", "+str(areaMikro_sigmal11.round(2))+ ", "+ str(areaMikro_standardfehlerl11.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel11.round(2))+", " +str(perimeterMikro_variancel11.round(2))+ ", "+str(perimeterMikro_sigmal11.round(2))+ ", "+ str(perimeterMikro_standardfehlerl11.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel11.round(2))+", " + str(lengthMikro_variancel11.round(2))+ ", "+str(lengthMikro_sigmal11.round(2))+ ", "+ str(lengthMikro_standardfehlerl11.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel11.round(2))+", " + str(aMikro_variancel11.round(2))+ ", "+str(aMikro_sigmal11.round(2))+ ", "+ str(aMikro_standardfehlerl11.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel11.round(2))+", " +str(bMikro_variancel11.round(2))+ ", "+str(bMikro_sigmal11.round(2))+ ", "+ str(bMikro_standardfehlerl11.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel11.round(2)) +", "+ str(a2Mikro_variancel11.round(2))+ ", "+str(a2Mikro_sigmal11.round(2))+ ", "+ str(a2Mikro_standardfehlerl11.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel11.round(2)) +", "+ str(b2Mikro_variancel11.round(2))+ ", "+str(b2Mikro_sigmal11.round(2))+ ", "+ str(b2Mikro_standardfehlerl11.round(2))+"\n")
wf.write("Amount: " + str(icountl11) + "\n"+ "\n")


wf.write(str(l11Micro)+ " to " + str(l12Micro)+"\n")

wf.write("Area in micrometer: " + str(areaMikro_averagel12.round(2))+", " + str(areaMikro_variancel12.round(2))+ ", "+str(areaMikro_sigmal12.round(2))+ ", "+ str(areaMikro_standardfehlerl12.round(2))+"\n")
wf.write("Perimeter in micrometer: " + str(perimeterMikro_averagel12.round(2))+", " +str(perimeterMikro_variancel12.round(2))+ ", "+str(perimeterMikro_sigmal12.round(2))+ ", "+ str(perimeterMikro_standardfehlerl12.round(2))+ "\n")
wf.write("Length in micrometer: " + str(lengthMikro_averagel12.round(2))+", " + str(lengthMikro_variancel12.round(2))+ ", "+str(lengthMikro_sigmal12.round(2))+ ", "+ str(lengthMikro_standardfehlerl12.round(2))+"\n")
wf.write("a in micrometer: " + str(aMikro_averagel12.round(2))+", " + str(aMikro_variancel12.round(2))+ ", "+str(aMikro_sigmal12.round(2))+ ", "+ str(aMikro_standardfehlerl12.round(2))+"\n")
wf.write("b in micrometer: " + str( bMikro_averagel12.round(2)) +", "+str(bMikro_variancel12.round(2))+ ", "+str(bMikro_sigmal12.round(2))+ ", "+ str(bMikro_standardfehlerl12.round(2))+ "\n")
wf.write("a2 in micrometer: " + str(a2Mikro_averagel12.round(2)) +", "+ str(a2Mikro_variancel12.round(2))+ ", "+str(a2Mikro_sigmal12.round(2))+ ", "+ str(a2Mikro_standardfehlerl12.round(2))+"\n")
wf.write("b2 in micrometer: " + str(b2Mikro_averagel12.round(2)) +", "+ str(b2Mikro_variancel12.round(2))+ ", "+str(b2Mikro_sigmal12.round(2))+ ", "+ str(b2Mikro_standardfehlerl12.round(2))+"\n")
wf.write("Amount: " + str(icountl12) + "\n"+ "\n")


wf.write("Amount: " + str(icount))



#####################Splines########################

###############l1

allcellsPosl1T = allcellsPosl1.transpose()

for row2 in range(np.size(allcellsPosl1T,0)):
	maxval = np.amax(allcellsPosl1T[row2])
	for value in range(np.size(allcellsPosl1T,1)):	
		if allcellsPosl1T[row2][value]>=maxval: 
			allcellsPosl1Spline[value][row2]=allcellsPosl1T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl1FH[ent][ent2] = allcellsPosl1[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl1RH[ent][ent2] = allcellsPosl1[ent][ent2] 

for row2 in range(np.size(allcellsl1FH,0)):
	maxval = np.amax(allcellsl1FH[row2])
	for value in range(np.size(allcellsl1FH,1)):	
		if allcellsl1FH[row2][value]>=maxval: 
			allcellsl1FHSpl[row2][value]=allcellsl1FH[row2][value]

for row2 in range(np.size(allcellsl1RH,0)):
	maxval = np.amax(allcellsl1RH[row2])
	for value in range(np.size(allcellsl1RH,1)):	
		if allcellsl1RH[row2][value]>=maxval: 
			allcellsl1RHSpl[row2][value]=allcellsl1RH[row2][value]

BothSplines1 = allcellsl1RHSpl + allcellsl1FHSpl

ispl = max_y/2
mspl = 0


BothFinl1 = allcellsPosl1Spline + BothSplines1

MaxInt = np.amax(BothFinl1)

#MakeImagetrMicro(BothFinl1, "Cells_SplinePrep "+str(l6Micro) + " to " + str(l1Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l6Micro) + "_" + str(l1Micro))

for row2 in range(np.size(BothFinl1,0)): 
	for value in range(np.size(BothFinl1,1)): 
		if BothFinl1[row2][value]<MaxInt*0.3: 
			BothFinl1[row2][value] = 0	

#MakeImagetrMicro(BothFinl1, "Cells_SplinePrepRed_ "+str(l6Micro) + " to " + str(l1Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l6Micro) + "_" + str(l1Micro))

Splinepointsl1 = []

for val in range(np.size(BothFinl1,1)): 
	for row in range(np.size(BothFinl1,0)): 
		if BothFinl1[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl1.append([row,val])
xlistl1 = []
ylistl1 = []
print Splinepointsl1
for i in Splinepointsl1: 
	xlistl1.append(i[0])
	ylistl1.append(i[1])
testp = []
testp = [xlistl1,ylistl1]
#plt.plot(ylistl1,xlistl1,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl1,ImageLocOut + "Micrometer" + "/_Splinel1")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()

###############l2

allcellsPosl2T = allcellsPosl2.transpose()

for row2 in range(np.size(allcellsPosl2T,0)):
	maxval = np.amax(allcellsPosl2T[row2])
	for value in range(np.size(allcellsPosl2T,1)):	
		if allcellsPosl2T[row2][value]>=maxval: 
			allcellsPosl2Spline[value][row2]=allcellsPosl2T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl2FH[ent][ent2] = allcellsPosl2[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl2RH[ent][ent2] = allcellsPosl2[ent][ent2] 

for row2 in range(np.size(allcellsl2FH,0)):
	maxval = np.amax(allcellsl2FH[row2])
	for value in range(np.size(allcellsl2FH,1)):	
		if allcellsl2FH[row2][value]>=maxval: 
			allcellsl2FHSpl[row2][value]=allcellsl2FH[row2][value]

for row2 in range(np.size(allcellsl2RH,0)):
	maxval = np.amax(allcellsl2RH[row2])
	for value in range(np.size(allcellsl2RH,1)):	
		if allcellsl2RH[row2][value]>=maxval: 
			allcellsl2RHSpl[row2][value]=allcellsl2RH[row2][value]

BothSplines2 = allcellsl2RHSpl + allcellsl2FHSpl

ispl = max_y/2
mspl = 0


BothFinl2 = allcellsPosl2Spline + BothSplines2

MaxInt = np.amax(BothFinl2)

#MakeImagetrMicro(BothFinl2, "Cells_SplinePrep "+str(l6Micro) + " to " + str(l2Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l6Micro) + "_" + str(l2Micro))

for row2 in range(np.size(BothFinl2,0)): 
	for value in range(np.size(BothFinl2,1)): 
		if BothFinl2[row2][value]<MaxInt*0.3: 
			BothFinl2[row2][value] = 0	

#MakeImagetrMicro(BothFinl2, "Cells_SplinePrepRed_ "+str(l6Micro) + " to " + str(l2Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l6Micro) + "_" + str(l2Micro))

Splinepointsl2 = []

for val in range(np.size(BothFinl2,1)): 
	for row in range(np.size(BothFinl2,0)): 
		if BothFinl2[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl2.append([row,val])
xlistl2 = []
ylistl2 = []
print Splinepointsl2
for i in Splinepointsl2: 
	xlistl2.append(i[0])
	ylistl2.append(i[1])
testp = []
testp = [xlistl2,ylistl2]
#plt.plot(ylistl2,xlistl2,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl2,ImageLocOut + "Micrometer" + "/_Splinel2")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()

###############l3

allcellsPosl3T = allcellsPosl3.transpose()

for row2 in range(np.size(allcellsPosl3T,0)):
	maxval = np.amax(allcellsPosl3T[row2])
	for value in range(np.size(allcellsPosl3T,1)):	
		if allcellsPosl3T[row2][value]>=maxval: 
			allcellsPosl3Spline[value][row2]=allcellsPosl3T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl3FH[ent][ent2] = allcellsPosl3[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl3RH[ent][ent2] = allcellsPosl3[ent][ent2] 

for row2 in range(np.size(allcellsl3FH,0)):
	maxval = np.amax(allcellsl3FH[row2])
	for value in range(np.size(allcellsl3FH,1)):	
		if allcellsl3FH[row2][value]>=maxval: 
			allcellsl3FHSpl[row2][value]=allcellsl3FH[row2][value]

for row2 in range(np.size(allcellsl3RH,0)):
	maxval = np.amax(allcellsl3RH[row2])
	for value in range(np.size(allcellsl3RH,1)):	
		if allcellsl3RH[row2][value]>=maxval: 
			allcellsl3RHSpl[row2][value]=allcellsl3RH[row2][value]

BothSplines3 = allcellsl3RHSpl + allcellsl3FHSpl

ispl = max_y/2
mspl = 0


BothFinl3 = allcellsPosl3Spline + BothSplines3

MaxInt = np.amax(BothFinl3)

#MakeImagetrMicro(BothFinl3, "Cells_SplinePrep "+str(l6Micro) + " to " + str(l3Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l6Micro) + "_" + str(l3Micro))

for row2 in range(np.size(BothFinl3,0)): 
	for value in range(np.size(BothFinl3,1)): 
		if BothFinl3[row2][value]<MaxInt*0.3: 
			BothFinl3[row2][value] = 0	

#MakeImagetrMicro(BothFinl3, "Cells_SplinePrepRed_ "+str(l6Micro) + " to " + str(l3Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l6Micro) + "_" + str(l3Micro))

Splinepointsl3 = []

for val in range(np.size(BothFinl3,1)): 
	for row in range(np.size(BothFinl3,0)): 
		if BothFinl3[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl3.append([row,val])
xlistl3 = []
ylistl3 = []
print Splinepointsl3
for i in Splinepointsl3: 
	xlistl3.append(i[0])
	ylistl3.append(i[1])
testp = []
testp = [xlistl3,ylistl3]
#plt.plot(ylistl3,xlistl3,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl3,ImageLocOut + "Micrometer" + "/_Splinel3")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()

###############l4

allcellsPosl4T = allcellsPosl4.transpose()

for row2 in range(np.size(allcellsPosl4T,0)):
	maxval = np.amax(allcellsPosl4T[row2])
	for value in range(np.size(allcellsPosl4T,1)):	
		if allcellsPosl4T[row2][value]>=maxval: 
			allcellsPosl4Spline[value][row2]=allcellsPosl4T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl4FH[ent][ent2] = allcellsPosl4[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl4RH[ent][ent2] = allcellsPosl4[ent][ent2] 

for row2 in range(np.size(allcellsl4FH,0)):
	maxval = np.amax(allcellsl4FH[row2])
	for value in range(np.size(allcellsl4FH,1)):	
		if allcellsl4FH[row2][value]>=maxval: 
			allcellsl4FHSpl[row2][value]=allcellsl4FH[row2][value]

for row2 in range(np.size(allcellsl4RH,0)):
	maxval = np.amax(allcellsl4RH[row2])
	for value in range(np.size(allcellsl4RH,1)):	
		if allcellsl4RH[row2][value]>=maxval: 
			allcellsl4RHSpl[row2][value]=allcellsl4RH[row2][value]

BothSplines4 = allcellsl4RHSpl + allcellsl4FHSpl

ispl = max_y/2
mspl = 0


BothFinl4 = allcellsPosl4Spline + BothSplines4

MaxInt = np.amax(BothFinl4)

#MakeImagetrMicro(BothFinl4, "Cells_SplinePrep "+str(l6Micro) + " to " + str(l4Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l6Micro) + "_" + str(l4Micro))

for row2 in range(np.size(BothFinl4,0)): 
	for value in range(np.size(BothFinl4,1)): 
		if BothFinl4[row2][value]<MaxInt*0.3: 
			BothFinl4[row2][value] = 0	

#MakeImagetrMicro(BothFinl4, "Cells_SplinePrepRed_ "+str(l6Micro) + " to " + str(l4Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l6Micro) + "_" + str(l4Micro))

Splinepointsl4 = []

for val in range(np.size(BothFinl4,1)): 
	for row in range(np.size(BothFinl4,0)): 
		if BothFinl4[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl4.append([row,val])
xlistl4 = []
ylistl4 = []
print Splinepointsl4
for i in Splinepointsl4: 
	xlistl4.append(i[0])
	ylistl4.append(i[1])
testp = []
testp = [xlistl4,ylistl4]
#plt.plot(ylistl4,xlistl4,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl4,ImageLocOut + "Micrometer" + "/_Splinel4")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()



#################l5


allcellsPosl5T = allcellsPosl5.transpose()

for row2 in range(np.size(allcellsPosl5T,0)):
	maxval = np.amax(allcellsPosl5T[row2])
	for value in range(np.size(allcellsPosl5T,1)):	
		if allcellsPosl5T[row2][value]>=maxval: 
			allcellsPosl5Spline[value][row2]=allcellsPosl5T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl5FH[ent][ent2] = allcellsPosl5[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl5RH[ent][ent2] = allcellsPosl5[ent][ent2] 

for row2 in range(np.size(allcellsl5FH,0)):
	maxval = np.amax(allcellsl5FH[row2])
	for value in range(np.size(allcellsl5FH,1)):	
		if allcellsl5FH[row2][value]>=maxval: 
			allcellsl5FHSpl[row2][value]=allcellsl5FH[row2][value]

for row2 in range(np.size(allcellsl5RH,0)):
	maxval = np.amax(allcellsl5RH[row2])
	for value in range(np.size(allcellsl5RH,1)):	
		if allcellsl5RH[row2][value]>=maxval: 
			allcellsl5RHSpl[row2][value]=allcellsl5RH[row2][value]

BothSplines5 = allcellsl5RHSpl + allcellsl5FHSpl

ispl = max_y/2
mspl = 0


BothFinl5 = allcellsPosl5Spline + BothSplines5

MaxInt = np.amax(BothFinl5)

#MakeImagetrMicro(BothFinl5, "Cells_SplinePrep "+str(l4Micro) + " to " + str(l5Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l4Micro) + "_" + str(l5Micro))

for row2 in range(np.size(BothFinl5,0)): 
	for value in range(np.size(BothFinl5,1)): 
		if BothFinl5[row2][value]<MaxInt*0.6: 
			BothFinl5[row2][value] = 0	

#MakeImagetrMicro(BothFinl5, "Cells_SplinePrepRed_ "+str(l4Micro) + " to " + str(l5Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l4Micro) + "_" + str(l5Micro))

Splinepointsl5 = []

for val in range(np.size(BothFinl5,1)): 
	for row in range(np.size(BothFinl5,0)): 
		if BothFinl5[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl5.append([row,val])
xlistl5 = []
ylistl5 = []
print Splinepointsl5
for i in Splinepointsl5: 
	xlistl5.append(i[0])
	ylistl5.append(i[1])
testp = []
testp = [xlistl5,ylistl5]
print "xlist: ", xlistl5
print "ylist: ", ylistl5
#plt.plot(ylistl5,xlistl5,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl5,ImageLocOut + "Micrometer" +"/" + "_Splinel5")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()

###############l6

allcellsPosl6T = allcellsPosl6.transpose()

for row2 in range(np.size(allcellsPosl6T,0)):
	maxval = np.amax(allcellsPosl6T[row2])
	for value in range(np.size(allcellsPosl6T,1)):	
		if allcellsPosl6T[row2][value]>=maxval: 
			allcellsPosl6Spline[value][row2]=allcellsPosl6T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl6FH[ent][ent2] = allcellsPosl6[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl6RH[ent][ent2] = allcellsPosl6[ent][ent2] 

for row2 in range(np.size(allcellsl6FH,0)):
	maxval = np.amax(allcellsl6FH[row2])
	for value in range(np.size(allcellsl6FH,1)):	
		if allcellsl6FH[row2][value]>=maxval: 
			allcellsl6FHSpl[row2][value]=allcellsl6FH[row2][value]

for row2 in range(np.size(allcellsl6RH,0)):
	maxval = np.amax(allcellsl6RH[row2])
	for value in range(np.size(allcellsl6RH,1)):	
		if allcellsl6RH[row2][value]>=maxval: 
			allcellsl6RHSpl[row2][value]=allcellsl6RH[row2][value]

BothSplines6 = allcellsl6RHSpl + allcellsl6FHSpl

ispl = max_y/2
mspl = 0


BothFinl6 = allcellsPosl6Spline + BothSplines6

MaxInt = np.amax(BothFinl6)

#MakeImagetrMicro(BothFinl6, "Cells_SplinePrep "+str(l5Micro) + " to " + str(l6Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l5Micro) + "_" + str(l6Micro))

for row2 in range(np.size(BothFinl6,0)): 
	for value in range(np.size(BothFinl6,1)): 
		if BothFinl6[row2][value]<MaxInt*0.5: 
			BothFinl6[row2][value] = 0	

MakeImagetrMicro(BothFinl6, "Cells_SplinePrepRed_ "+str(l5Micro) + " to " + str(l6Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l5Micro) + "_" + str(l6Micro))

Splinepointsl6 = []

for val in range(np.size(BothFinl6,1)): 
	for row in range(np.size(BothFinl6,0)): 
		if BothFinl6[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl6.append([row,val])
xlistl6 = []
ylistl6 = []
print Splinepointsl6
for i in Splinepointsl6: 
	xlistl6.append(i[0])
	ylistl6.append(i[1])
testp = []
testp = [xlistl6,ylistl6]
print "xlist: ", xlistl6
print "ylist: ", ylistl6
#plt.plot(ylistl6,xlistl6,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl6,ImageLocOut + "Micrometer" + "/_Splinel6")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()


###############l7

allcellsPosl7T = allcellsPosl7.transpose()

for row2 in range(np.size(allcellsPosl7T,0)):
	maxval = np.amax(allcellsPosl7T[row2])
	for value in range(np.size(allcellsPosl7T,1)):	
		if allcellsPosl7T[row2][value]>=maxval: 
			allcellsPosl7Spline[value][row2]=allcellsPosl7T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl7FH[ent][ent2] = allcellsPosl7[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl7RH[ent][ent2] = allcellsPosl7[ent][ent2] 

for row2 in range(np.size(allcellsl7FH,0)):
	maxval = np.amax(allcellsl7FH[row2])
	for value in range(np.size(allcellsl7FH,1)):	
		if allcellsl7FH[row2][value]>=maxval: 
			allcellsl7FHSpl[row2][value]=allcellsl7FH[row2][value]

for row2 in range(np.size(allcellsl7RH,0)):
	maxval = np.amax(allcellsl7RH[row2])
	for value in range(np.size(allcellsl7RH,1)):	
		if allcellsl7RH[row2][value]>=maxval: 
			allcellsl7RHSpl[row2][value]=allcellsl7RH[row2][value]

BothSplines7 = allcellsl7RHSpl + allcellsl7FHSpl

ispl = max_y/2
mspl = 0


BothFinl7 = allcellsPosl7Spline + BothSplines7

MaxInt = np.amax(BothFinl7)

#MakeImagetrMicro(BothFinl7, "Cells_SplinePrep "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l6Micro) + "_" + str(l7Micro))

for row2 in range(np.size(BothFinl7,0)): 
	for value in range(np.size(BothFinl7,1)): 
		if BothFinl7[row2][value]<MaxInt*0.3: 
			BothFinl7[row2][value] = 0	

#MakeImagetrMicro(BothFinl7, "Cells_SplinePrepRed_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l6Micro) + "_" + str(l7Micro))

Splinepointsl7 = []

for val in range(np.size(BothFinl7,1)): 
	for row in range(np.size(BothFinl7,0)): 
		if BothFinl7[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl7.append([row,val])
xlistl7 = []
ylistl7 = []
print Splinepointsl7
for i in Splinepointsl7: 
	xlistl7.append(i[0])
	ylistl7.append(i[1])
testp = []
testp = [xlistl7,ylistl7]
#plt.plot(ylistl7,xlistl7,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl7,ImageLocOut + "Micrometer" + "/_Splinel7")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()


###############l8


allcellsPosl8T = allcellsPosl8.transpose()

for row2 in range(np.size(allcellsPosl8T,0)):
	maxval = np.amax(allcellsPosl8T[row2])
	for value in range(np.size(allcellsPosl8T,1)):	
		if allcellsPosl8T[row2][value]>=maxval: 
			allcellsPosl8Spline[value][row2]=allcellsPosl8T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl8FH[ent][ent2] = allcellsPosl8[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl8RH[ent][ent2] = allcellsPosl8[ent][ent2] 

for row2 in range(np.size(allcellsl8FH,0)):
	maxval = np.amax(allcellsl8FH[row2])
	for value in range(np.size(allcellsl8FH,1)):	
		if allcellsl8FH[row2][value]>=maxval: 
			allcellsl8FHSpl[row2][value]=allcellsl8FH[row2][value]

for row2 in range(np.size(allcellsl8RH,0)):
	maxval = np.amax(allcellsl8RH[row2])
	for value in range(np.size(allcellsl8RH,1)):	
		if allcellsl8RH[row2][value]>=maxval: 
			allcellsl8RHSpl[row2][value]=allcellsl8RH[row2][value]

BothSplines8 = allcellsl8RHSpl + allcellsl8FHSpl

ispl = max_y/2
mspl = 0


BothFinl8 = allcellsPosl8Spline + BothSplines8

MaxInt = np.amax(BothFinl8)

MakeImagetrMicro(BothFinl8, "Cells_SplinePrep "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l7Micro) + "_" + str(l8Micro))

for row2 in range(np.size(BothFinl8,0)): 
	for value in range(np.size(BothFinl8,1)): 
		if BothFinl8[row2][value]<MaxInt*0.45: #0.4 0.5
			BothFinl8[row2][value] = 0	

MakeImagetrMicro(BothFinl8, "Cells_SplinePrepRed_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l7Micro) + "_" + str(l8Micro))

Splinepointsl8 = []
WeightArray = []
WeightArray2 = []
print "BothFinl8: ", BothFinl8
print "Maxint: ", MaxInt
for val in range(np.size(BothFinl8,1)): 
	for row in range(np.size(BothFinl8,0)): 
		if BothFinl8[row][val] > 0: 
			row2 = row-(y0_c-1)
			val2 = val - (x0_c-1)
			Splinepointsl8.append([row2,val2])
			#if BothFinl8[row][val] > MaxInt*0.8:
			#	WeightArray2.append(3)
			#elif BothFinl8[row][val] > MaxInt*0.6: 
			#	WeightArray2.append(2)
			#else: 
			#	WeightArray2.append(1)
			if BothFinl8[row][val] > MaxInt*0.7:
				WeightArray2.append(2)
			else: 
				WeightArray2.append(1)
			WeightArray.append(BothFinl8[row][val])
xlistl8 = []
ylistl8 = []
print "WArray: ", WeightArray
print "WArray2: ", WeightArray2
#print Splinepointsl8
for i in Splinepointsl8: 
	xlistl8.append(i[0])
	ylistl8.append(i[1])
testp = []
testp = [xlistl8,ylistl8]
plt.plot(ylistl8,xlistl8,'ro')
plt.show()
spline = Spline_Interpolation(Splinepointsl8,WeightArray2,ImageLocOut + "Micrometer" + "/_Splinel8_Defense")
spline.show_naturalshape()
spline.show_naturalshapeMicro()

###############################

allcellsPosl8T = allcellsPosl8.transpose()

for row2 in range(np.size(allcellsPosl8T,0)):
	maxval = np.amax(allcellsPosl8T[row2])
	for value in range(np.size(allcellsPosl8T,1)):	
		if allcellsPosl8T[row2][value]>=maxval: 
			allcellsPosl8Spline[value][row2]=allcellsPosl8T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl8FH[ent][ent2] = allcellsPosl8[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl8RH[ent][ent2] = allcellsPosl8[ent][ent2] 

for row2 in range(np.size(allcellsl8FH,0)):
	maxval = np.amax(allcellsl8FH[row2])
	for value in range(np.size(allcellsl8FH,1)):	
		if allcellsl8FH[row2][value]>=maxval: 
			allcellsl8FHSpl[row2][value]=allcellsl8FH[row2][value]

for row2 in range(np.size(allcellsl8RH,0)):
	maxval = np.amax(allcellsl8RH[row2])
	for value in range(np.size(allcellsl8RH,1)):	
		if allcellsl8RH[row2][value]>=maxval: 
			allcellsl8RHSpl[row2][value]=allcellsl8RH[row2][value]

BothSplines8 = allcellsl8RHSpl + allcellsl8FHSpl

ispl = max_y/2
mspl = 0


BothFinl8 = allcellsPosl8Spline + BothSplines8

MaxInt = np.amax(BothFinl8)

MakeImagetrMicro(BothFinl8, "Cells_SplinePrep "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l7Micro) + "_" + str(l8Micro))

for row2 in range(np.size(BothFinl8,0)): 
	for value in range(np.size(BothFinl8,1)): 
		if BothFinl8[row2][value]<MaxInt*0.45: #0.4 0.5
			BothFinl8[row2][value] = 0	

MakeImagetrMicro(BothFinl8, "Cells_SplinePrepRed_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l7Micro) + "_" + str(l8Micro))

Splinepointsl8 = []
WeightArray = []
WeightArray2 = []
print "BothFinl8: ", BothFinl8
print "Maxint: ", MaxInt
for val in range(np.size(BothFinl8,1)): 
	for row in range(np.size(BothFinl8,0)): 
		if BothFinl8[row][val] > 0: 
			row2 = row-(y0_c-1)
			val2 = val - (x0_c-1)
			Splinepointsl8.append([row2,val2])
			#if BothFinl8[row][val] > MaxInt*0.8:
			#	WeightArray2.append(3)
			#elif BothFinl8[row][val] > MaxInt*0.6: 
			#	WeightArray2.append(2)
			#else: 
			#	WeightArray2.append(1)
			if BothFinl8[row][val] > MaxInt*0.7:
				WeightArray2.append(2)
			else: 
				WeightArray2.append(1)
			WeightArray.append(BothFinl8[row][val])
xlistl8 = []
ylistl8 = []
print "WArray: ", WeightArray
print "WArray2: ", WeightArray2
#print Splinepointsl8
for i in Splinepointsl8: 
	xlistl8.append(i[0])
	ylistl8.append(i[1])
testp = []
testp = [xlistl8,ylistl8]
plt.plot(ylistl8,xlistl8,'ro')
plt.show()
spline = Spline_Interpolation(Splinepointsl8,WeightArray2,ImageLocOut + "Micrometer" + "/_Splinel8")
spline.show_naturalshape()
spline.show_naturalshapeMicro()



###############l9

allcellsPosl9T = allcellsPosl9.transpose()

for row2 in range(np.size(allcellsPosl9T,0)):
	maxval = np.amax(allcellsPosl9T[row2])
	for value in range(np.size(allcellsPosl9T,1)):	
		if allcellsPosl9T[row2][value]>=maxval: 
			allcellsPosl9Spline[value][row2]=allcellsPosl9T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl9FH[ent][ent2] = allcellsPosl9[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl9RH[ent][ent2] = allcellsPosl9[ent][ent2] 

for row2 in range(np.size(allcellsl9FH,0)):
	maxval = np.amax(allcellsl9FH[row2])
	for value in range(np.size(allcellsl9FH,1)):	
		if allcellsl9FH[row2][value]>=maxval: 
			allcellsl9FHSpl[row2][value]=allcellsl9FH[row2][value]

for row2 in range(np.size(allcellsl9RH,0)):
	maxval = np.amax(allcellsl9RH[row2])
	for value in range(np.size(allcellsl9RH,1)):	
		if allcellsl9RH[row2][value]>=maxval: 
			allcellsl9RHSpl[row2][value]=allcellsl9RH[row2][value]

BothSplines9 = allcellsl9RHSpl + allcellsl9FHSpl

ispl = max_y/2
mspl = 0


BothFinl9 = allcellsPosl9Spline + BothSplines9

MaxInt = np.amax(BothFinl9)

MakeImagetrMicro(BothFinl9, "Cells_SplinePrep "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l8Micro) + "_" + str(l9Micro))

for row2 in range(np.size(BothFinl9,0)): 
	for value in range(np.size(BothFinl9,1)): 
		if BothFinl9[row2][value]<MaxInt*0.3: 
			BothFinl9[row2][value] = 0	

MakeImagetrMicro(BothFinl9, "Cells_SplinePrepRed_ "+str(l9Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l8Micro) + "_" + str(l9Micro))

Splinepointsl9 = []

for val in range(np.size(BothFinl9,1)): 
	for row in range(np.size(BothFinl9,0)): 
		if BothFinl9[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl9.append([row,val])
xlistl9 = []
ylistl9 = []
print Splinepointsl9
for i in Splinepointsl9: 
	xlistl9.append(i[0])
	ylistl9.append(i[1])
testp = []
testp = [xlistl9,ylistl9]
#plt.plot(ylistl9,xlistl9,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl9,ImageLocOut + "Micrometer" + "/_Splinel9")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()



###############l10

allcellsPosl10T = allcellsPosl10.transpose()

for row2 in range(np.size(allcellsPosl10T,0)):
	maxval = np.amax(allcellsPosl10T[row2])
	for value in range(np.size(allcellsPosl10T,1)):	
		if allcellsPosl10T[row2][value]>=maxval: 
			allcellsPosl10Spline[value][row2]=allcellsPosl10T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl10FH[ent][ent2] = allcellsPosl10[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl10RH[ent][ent2] = allcellsPosl10[ent][ent2] 

for row2 in range(np.size(allcellsl10FH,0)):
	maxval = np.amax(allcellsl10FH[row2])
	for value in range(np.size(allcellsl10FH,1)):	
		if allcellsl10FH[row2][value]>=maxval: 
			allcellsl10FHSpl[row2][value]=allcellsl10FH[row2][value]

for row2 in range(np.size(allcellsl10RH,0)):
	maxval = np.amax(allcellsl10RH[row2])
	for value in range(np.size(allcellsl7RH,1)):	
		if allcellsl10RH[row2][value]>=maxval: 
			allcellsl10RHSpl[row2][value]=allcellsl10RH[row2][value]

BothSplines10 = allcellsl10RHSpl + allcellsl10FHSpl

ispl = max_y/2
mspl = 0


BothFinl10 = allcellsPosl10Spline + BothSplines10

MaxInt = np.amax(BothFinl10)

MakeImagetrMicro(BothFinl10, "Cells_SplinePrep "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l9Micro) + "_" + str(l10Micro))

for row2 in range(np.size(BothFinl10,0)): 
	for value in range(np.size(BothFinl10,1)): 
		if BothFinl10[row2][value]<MaxInt*0.3: 
			BothFinl10[row2][value] = 0	

MakeImagetrMicro(BothFinl10, "Cells_SplinePrepRed_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l9Micro) + "_" + str(l10Micro))

Splinepointsl10 = []

#for val in range(np.size(BothFinl10,1)): 
#	for row in range(np.size(BothFinl10,0)): 
#		if BothFinl10[row][val] > 0: 
#			row = row-y0_c
#			val = val - x0_c
#			Splinepointsl10.append([row,val])
#xlistl10 = []
#ylistl10 = []
#print Splinepointsl10
#for i in Splinepointsl10: 
#	xlistl10.append(i[0])
#	ylistl10.append(i[1])
#testp = []
#testp = [xlistl10,ylistl10]
#plt.plot(ylistl10,xlistl10,'ro')
#plt.show()
#spline = Spline_Interpolation(Splinepointsl10,ImageLocOut + "Micrometer" + "/_Splinel10")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()



###############l11

allcellsPosl11T = allcellsPosl11.transpose()

for row2 in range(np.size(allcellsPosl11T,0)):
	maxval = np.amax(allcellsPosl11T[row2])
	for value in range(np.size(allcellsPosl11T,1)):	
		if allcellsPosl11T[row2][value]>=maxval: 
			allcellsPosl11Spline[value][row2]=allcellsPosl11T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl11FH[ent][ent2] = allcellsPosl11[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl11RH[ent][ent2] = allcellsPosl11[ent][ent2] 

for row2 in range(np.size(allcellsl11FH,0)):
	maxval = np.amax(allcellsl11FH[row2])
	for value in range(np.size(allcellsl11FH,1)):	
		if allcellsl11FH[row2][value]>=maxval: 
			allcellsl11FHSpl[row2][value]=allcellsl11FH[row2][value]

for row2 in range(np.size(allcellsl11RH,0)):
	maxval = np.amax(allcellsl11RH[row2])
	for value in range(np.size(allcellsl11RH,1)):	
		if allcellsl11RH[row2][value]>=maxval: 
			allcellsl11RHSpl[row2][value]=allcellsl11RH[row2][value]

BothSplines11 = allcellsl11RHSpl + allcellsl11FHSpl

ispl = max_y/2
mspl = 0


BothFinl11 = allcellsPosl11Spline + BothSplines11

MaxInt = np.amax(BothFinl11)

#MakeImagetrMicro(BothFinl11, "Cells_SplinePrep "+str(l6Micro) + " to " + str(l11Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l6Micro) + "_" + str(l11Micro))

for row2 in range(np.size(BothFinl11,0)): 
	for value in range(np.size(BothFinl11,1)): 
		if BothFinl11[row2][value]<MaxInt*0.3: 
			BothFinl11[row2][value] = 0	

#MakeImagetrMicro(BothFinl11, "Cells_SplinePrepRed_ "+str(l6Micro) + " to " + str(l11Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l6Micro) + "_" + str(l11Micro))

Splinepointsl11 = []

for val in range(np.size(BothFinl11,1)): 
	for row in range(np.size(BothFinl11,0)): 
		if BothFinl11[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl11.append([row,val])
xlistl11 = []
ylistl11 = []
print Splinepointsl11
for i in Splinepointsl11: 
	xlistl11.append(i[0])
	ylistl11.append(i[1])
testp = []
testp = [xlistl11,ylistl11]
plt.plot(ylistl11,xlistl11,'ro')
plt.show()
#spline = Spline_Interpolation(Splinepointsl11,ImageLocOut + "Micrometer" + "/_Splinel11")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()

###############l12

allcellsPosl12T = allcellsPosl12.transpose()

for row2 in range(np.size(allcellsPosl12T,0)):
	maxval = np.amax(allcellsPosl12T[row2])
	for value in range(np.size(allcellsPosl12T,1)):	
		if allcellsPosl12T[row2][value]>=maxval: 
			allcellsPosl12Spline[value][row2]=allcellsPosl12T[row2][value]


for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl12FH[ent][ent2] = allcellsPosl12[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl12RH[ent][ent2] = allcellsPosl12[ent][ent2] 

for row2 in range(np.size(allcellsl12FH,0)):
	maxval = np.amax(allcellsl12FH[row2])
	for value in range(np.size(allcellsl12FH,1)):	
		if allcellsl12FH[row2][value]>=maxval: 
			allcellsl12FHSpl[row2][value]=allcellsl12FH[row2][value]

for row2 in range(np.size(allcellsl12RH,0)):
	maxval = np.amax(allcellsl12RH[row2])
	for value in range(np.size(allcellsl12RH,1)):	
		if allcellsl12RH[row2][value]>=maxval: 
			allcellsl12RHSpl[row2][value]=allcellsl12RH[row2][value]

BothSplines12 = allcellsl12RHSpl + allcellsl12FHSpl

ispl = max_y/2
mspl = 0


BothFinl12 = allcellsPosl12Spline + BothSplines12

MaxInt = np.amax(BothFinl12)

#MakeImagetrMicro(BothFinl12, "Cells_SplinePrep "+str(l6Micro) + " to " + str(l12Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" + "/" + "_Cells_SplinePrep_"+str(l6Micro) + "_" + str(l12Micro))

for row2 in range(np.size(BothFinl12,0)): 
	for value in range(np.size(BothFinl12,1)): 
		if BothFinl12[row2][value]<MaxInt*0.3: 
			BothFinl12[row2][value] = 0	

#MakeImagetrMicro(BothFinl12, "Cells_SplinePrepRed_ "+str(l6Micro) + " to " + str(l12Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut + "Micrometer" +"/" + "_Cells_SplinePrepRed_"+str(l6Micro) + "_" + str(l12Micro))

Splinepointsl12 = []

for val in range(np.size(BothFinl12,1)): 
	for row in range(np.size(BothFinl12,0)): 
		if BothFinl12[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepointsl12.append([row,val])
xlistl12 = []
ylistl12 = []
print Splinepointsl12
for i in Splinepointsl12: 
	xlistl12.append(i[0])
	ylistl12.append(i[1])
testp = []
testp = [xlistl12,ylistl12]
plt.plot(ylistl12,xlistl12,'ro')
plt.show()
#spline = Spline_Interpolation(Splinepointsl12,ImageLocOut + "Micrometer" + "/_Splinel12")
#spline.show_naturalshape()
#spline.show_naturalshapeMicro()




























# while ispl >=  0: 
# 	while mspl < max_x:

# 		if BothSplines[ispl][mspl] >= MaxInt/2: 
# 			BothSplinesRed[ispl][mspl] = BothSplines[ispl][mspl]
# 		elif BothSplines5[ispl][mspl] >= MaxInt/2: 
# 				BothSplinesRed[ispl][mspl] = BothSplines5[ispl][mspl]
# 		mspl = mspl+1
# 	mspl = 0
# 	ispl = ispl-1

# mspl = 0
# ispl = max_y/2
# while ispl< max_y: 
# 	while mspl < max_x:

# 		if BothSplines[ispl][mspl] >= MaxInt/2: 
# 			BothSplinesRed[ispl][mspl] = BothSplines[ispl][mspl]
# 		elif BothSplines5[ispl][mspl] >= MaxInt/2: 
# 				BothSplinesRed[ispl][mspl] = BothSplines5[ispl][mspl]
# 		mspl = mspl+1
# 	mspl = 0
# 	ispl = ispl+1

for row2 in range(np.size(BothFin5,0)): 
	for value in range(np.size(BothFin5,1)): 
		if BothFin5[row2][value]<MaxInt*0.65: 
			BothFin5[row2][value] = 0	

#MakeImagetrMicro(BothFin5, "Cells_SplineFin3_ "+str(l4Micro) + " to " + str(l5Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_SplineFin3_"+str(l4Micro) + "_" + str(l5Micro))

for row2 in range(np.size(BothFin5,0)): 
	for value in range(np.size(BothFin5,1)): 
		if BothFin5[row2][value]<MaxInt*0.7: 
			BothFin5[row2][value] = 0	

#MakeImagetrMicro(BothFin5, "Cells_SplineFin4_ "+str(l4Micro) + " to " + str(l5Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_SplineFin4_"+str(l4Micro) + "_" + str(l5Micro))

for row2 in range(np.size(BothFin5,0)): 
	for value in range(np.size(BothFin5,1)): 
		if BothFin5[row2][value]<MaxInt*0.75: 
			BothFin5[row2][value] = 0	

#MakeImagetrMicro(BothFin5, "Cells_SplineFin5_ "+str(l4Micro) + " to " + str(l5Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_SplineFin5_"+str(l4Micro) + "_" + str(l5Micro))


###################l7#############################################

for ent in range(max_y/2): 
	for ent2 in range(max_x): 
		allcellsl7pos[ent][ent2] = allcellsl7[ent][ent2] 

allcellsl7posT = allcellsl7pos.transpose()

for row2 in range(np.size(allcellsl7posT,0)):
	maxval = np.amax(allcellsl7posT[row2])
	for value in range(np.size(allcellsl7posT,1)):	
		if allcellsl7posT[row2][value]>=maxval: 
			allcellsl7posSpline[value][row2]=allcellsl7posT[row2][value]

for ent in range(max_y/2,max_y): 
	for ent2 in range(max_x): 
		allcellsl7neg[ent][ent2] = allcellsl7[ent][ent2] 

allcellsl7negT = allcellsl7neg.transpose()

for row2 in range(np.size(allcellsl7negT,0)):
	maxval = np.amax(allcellsl7negT[row2])
#	print "row2: ", row2
#	print "maxval: ", maxval
	for value in range(np.size(allcellsl7negT,1)):	
		if allcellsl7negT[row2][value]>=maxval: 
#			print "yes if"
			allcellsl7negSpline[value][row2]=allcellsl7negT[row2][value]

BothSplines = allcellsl7posSpline + allcellsl7negSpline

for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl7FH[ent][ent2] = allcellsl7[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl7RH[ent][ent2] = allcellsl7[ent][ent2] 

for row2 in range(np.size(allcellsl7FH,0)):
	print "hi"
	#print "row: ", row2
	print "allcells: ", allcellsl7FH[row2]
	maxval = np.amax(allcellsl7FH[row2])
	print "max: ", max
	#pdb.set_trace()
	for value in range(np.size(allcellsl7FH,1)):	
		print "hu"
		#pdb.set_trace()
		if allcellsl7FH[row2][value]>=maxval: 
			allcellsl7FHSpl[row2][value]=allcellsl7FH[row2][value]
			#pdb.set_trace()

for row2 in range(np.size(allcellsl7RH,0)):
	maxval = np.amax(allcellsl7RH[row2])
	#pdb.set_trace()
	for value in range(np.size(allcellsl7RH,1)):	
		if allcellsl7RH[row2][value]>=maxval: 
			allcellsl7RHSpl[row2][value]=allcellsl7RH[row2][value]

BothSplines7 = allcellsl7RHSpl + allcellsl7FHSpl

ispl = max_y/2
MaxInt = np.amax(BothSplines7)

mspl = 0
while ispl >=  0: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed7[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines7[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed7[ispl][mspl] = BothSplines7[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl-1

mspl = 0
ispl = max_y/2
while ispl< max_y: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed7[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines7[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed7[ispl][mspl] = BothSplines7[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl+1



BothFin7 = BothSplines + BothSplines7 


#MakeImagetrMicro(BothFin7, "Cells_SplineFin0_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_SplineFin0_"+str(l6Micro) + "_" + str(l7Micro))


for row2 in range(np.size(BothFin7,0)): 
	for value in range(np.size(BothFin7,1)): 
		if BothFin7[row2][value]<MaxInt/2: 
			BothFin7[row2][value] = 0


print "Red"
#MakeImagetrMicro(BothSplinesRed7, "Cells_Spline_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_Spline_"+str(l6Micro) + "_" + str(l7Micro))

#MakeImagetrMicro(BothSplines, "Cells_Spline0_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_Spline0_"+str(l6Micro) + "_" + str(l7Micro))

#MakeImagetrMicro(BothSplines7, "Cells_Spline01_ "+str(l6Micro) + " to " + str(l7),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_Spline01_"+str(l6Micro) + "_" + str(l7))

#MakeImagetrMicro(BothFin7, "Cells_SplineFin1_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_SplineFin1_"+str(l6Micro) + "_" + str(l7Micro))

for row2 in range(np.size(BothFin7,0)): 
	for value in range(np.size(BothFin7,1)): 
		if BothFin7[row2][value]<MaxInt*0.6: 
			BothFin7[row2][value] = 0

MakeImagetrMicro(BothFin7, "Cells_SplineFin2_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin2_"+str(l6Micro) + "_" + str(l7Micro))

for row2 in range(np.size(BothFin7,0)): 
	for value in range(np.size(BothFin7,1)): 
		if BothFin7[row2][value]<MaxInt*0.65: 
			BothFin7[row2][value] = 0

MakeImagetrMicro(BothFin7, "Cells_SplineFin3_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin3_"+str(l6Micro) + "_" + str(l7Micro))

for row2 in range(np.size(BothFin7,0)): 
	for value in range(np.size(BothFin7,1)): 
		if BothFin7[row2][value]<MaxInt*0.7: 
			BothFin7[row2][value] = 0

MakeImagetrMicro(BothFin7, "Cells_SplineFin4_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin4_"+str(l6Micro) + "_" + str(l7Micro))

for row2 in range(np.size(BothFin7,0)): 
	for value in range(np.size(BothFin7,1)): 
		if BothFin7[row2][value]<MaxInt*0.85: 
			BothFin7[row2][value] = 0

MakeImagetrMicro(BothFin7, "Cells_SplineFin5_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin5_"+str(l6Micro) + "_" + str(l7Micro))

Splinepoints = []

for val in range(np.size(BothFin7,1)): 
	for row in range(y0_c): 
		if BothFin7[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepoints.append([row,val])
xlist = []
ylist = []
print Splinepoints
for i in Splinepoints: 
	xlist.append(i[0])
	ylist.append(i[1])
testp = []
testp = [xlist,ylist]
print "xlist: ", xlist
print "ylist: ", ylist
plt.plot(ylist,xlist,'ro')
plt.show()
spline = Spline_Interpolation(Splinepoints,ImageLocOut + "/Spline75")
spline.show_naturalshape()
spline.show_naturalshapeMicro()
for row2 in range(np.size(BothFin7,0)): 
	for value in range(np.size(BothFin7,1)): 
		if BothFin7[row2][value]<MaxInt*0.5: 
			BothFin7[row2][value] = 0

MakeImagetrMicro(BothFin7, "Cells_SplineFin6_ "+str(l6Micro) + " to " + str(l7Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin6_"+str(l6Micro) + "_" + str(l7Micro))

Splinepoints2 = []

for val in range(np.size(BothFin7,1)): 
	for row in range(y0_c): 
		if BothFin7[row][val] > 0: 
			row = row-y0_c
			val = val - x0_c
			Splinepoints2.append([row,val])
xlist = []
ylist = []
print Splinepoints2
for i in Splinepoints2: 
	xlist.append(i[0])
	ylist.append(i[1])
testp = []
testp = [xlist,ylist]
print "xlist: ", xlist
print "ylist: ", ylist
plt.plot(ylist,xlist,'ro')
plt.show()
spline = Spline_Interpolation(Splinepoints2,ImageLocOut + "/Spline76")

###################l8#############################################

for ent in range(max_y/2): 
	for ent2 in range(max_x): 
		allcellsl8pos[ent][ent2] = allcellsl8[ent][ent2] 

allcellsl8posT = allcellsl8pos.transpose()


for row2 in range(np.size(allcellsl8posT,0)):
	maxval = np.amax(allcellsl8posT[row2])
#	print "row2: ", row2
#	print "maxval: ", maxval
	for value in range(np.size(allcellsl8posT,1)):	
		if allcellsl8posT[row2][value]>=maxval: 
#			print "yes if"
			allcellsl8posSpline[value][row2]=allcellsl8posT[row2][value]

for ent in range(max_y/2,max_y): 
	for ent2 in range(max_x): 
		allcellsl8neg[ent][ent2] = allcellsl8[ent][ent2] 

allcellsl8negT = allcellsl8neg.transpose()

for row2 in range(np.size(allcellsl8negT,0)):
	maxval = np.amax(allcellsl8negT[row2])
	for value in range(np.size(allcellsl8negT,1)):	
		if allcellsl8negT[row2][value]>=maxval: 
			allcellsl8negSpline[value][row2]=allcellsl8negT[row2][value]

BothSplines = allcellsl8posSpline + allcellsl8negSpline

for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl8FH[ent][ent2] = allcellsl8[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl8RH[ent][ent2] = allcellsl8[ent][ent2] 

for row2 in range(np.size(allcellsl8FH,0)):
	maxval = np.amax(allcellsl8FH[row2])
	for value in range(np.size(allcellsl8FH,1)):	
		if allcellsl8FH[row2][value]>=maxval: 
			allcellsl8FHSpl[row2][value]=allcellsl8FH[row2][value]

for row2 in range(np.size(allcellsl8RH,0)):
	maxval = np.amax(allcellsl8RH[row2])
	for value in range(np.size(allcellsl8RH,1)):	
		if allcellsl8RH[row2][value]>=maxval: 
			allcellsl8RHSpl[row2][value]=allcellsl8RH[row2][value]

BothSplines8 = allcellsl8RHSpl + allcellsl8FHSpl

ispl = max_y/2
MaxInt = np.amax(BothSplines8)

mspl = 0
while ispl >=  0: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed8[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines8[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed8[ispl][mspl] = BothSplines8[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl-1

mspl = 0
ispl = max_y/2
while ispl< max_y: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed8[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines8[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed8[ispl][mspl] = BothSplines8[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl+1

BothFin8 = BothSplines + BothSplines8


#MakeImagetrMicro(BothFin8, "Cells_SplineFin0_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
#ShowImage(ImageLocOut +"/" + "_Cells_SplineFin0_"+str(l7Micro) + "_" + str(l8Micro))


for row2 in range(np.size(BothFin8,0)): 
	for value in range(np.size(BothFin8,1)): 
		if BothFin8[row2][value]<MaxInt/2: 
			BothFin8[row2][value] = 0

print "Red"
MakeImagetrMicro(BothSplinesRed8, "Cells_Spline_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline_"+str(l7Micro) + "_" + str(l8Micro))

MakeImagetrMicro(BothSplines, "Cells_Spline0_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline0_"+str(l7Micro) + "_" + str(l8Micro))

MakeImagetrMicro(BothSplines8, "Cells_Spline01_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline01_"+str(l7Micro) + "_" + str(l8Micro))

MakeImagetrMicro(BothFin8, "Cells_SplineFin1_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin1_"+str(l7Micro) + "_" + str(l8Micro))

for row2 in range(np.size(BothFin8,0)): 
	for value in range(np.size(BothFin8,1)): 
		if BothFin8[row2][value]<MaxInt*0.6: 
			BothFin8[row2][value] = 0

MakeImagetrMicro(BothFin8, "Cells_SplineFin2_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin2_"+str(l7Micro) + "_" + str(l8Micro))

for row2 in range(np.size(BothFin8,0)): 
	for value in range(np.size(BothFin8,1)): 
		if BothFin8[row2][value]<MaxInt*0.65: 
			BothFin8[row2][value] = 0

MakeImagetrMicro(BothFin8, "Cells_SplineFin3_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin3_"+str(l7Micro) + "_" + str(l8Micro))

for row2 in range(np.size(BothFin8,0)): 
	for value in range(np.size(BothFin8,1)): 
		if BothFin8[row2][value]<MaxInt*0.7: 
			BothFin8[row2][value] = 0

MakeImagetrMicro(BothFin8, "Cells_SplineFin4_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin4_"+str(l7Micro) + "_" + str(l8Micro))


#Cellfile = open(Matlabfile + "/" + str(l7Micro) + "_" + str(l8Micro) + "_Cells.txt","w")
#for cell in BothFin8: 
#	cell2 = str(cell)[1:-1]
#	#print cell
#	#print cell2
#	Cellfile.write(cell2+"\n")

for row2 in range(np.size(BothFin8,0)): 
	for value in range(np.size(BothFin8,1)): 
		if BothFin8[row2][value]<MaxInt*0.75: 
			BothFin8[row2][value] = 0

MakeImagetrMicro(BothFin8, "Cells_SplineFin5_ "+str(l7Micro) + " to " + str(l8Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin5_"+str(l7Micro) + "_" + str(l8Micro))


###################l9#############################################

for ent in range(max_y/2): 
	for ent2 in range(max_x): 
		allcellsl9pos[ent][ent2] = allcellsl9[ent][ent2] 

allcellsl9posT = allcellsl9pos.transpose()


for row2 in range(np.size(allcellsl9posT,0)):
	maxval = np.amax(allcellsl9posT[row2])
	for value in range(np.size(allcellsl9posT,1)):	
		if allcellsl9posT[row2][value]>=maxval: 
			allcellsl9posSpline[value][row2]=allcellsl9posT[row2][value]

for ent in range(max_y/2,max_y): 
	for ent2 in range(max_x): 
		allcellsl9neg[ent][ent2] = allcellsl9[ent][ent2] 

allcellsl9negT = allcellsl9neg.transpose()

for row2 in range(np.size(allcellsl9negT,0)):
	maxval = np.amax(allcellsl9negT[row2])
	for value in range(np.size(allcellsl9negT,1)):	
		if allcellsl9negT[row2][value]>=maxval: 
			allcellsl9negSpline[value][row2]=allcellsl9negT[row2][value]

BothSplines = allcellsl9posSpline + allcellsl9negSpline

for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl9FH[ent][ent2] = allcellsl9[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl9RH[ent][ent2] = allcellsl9[ent][ent2] 

for row2 in range(np.size(allcellsl9FH,0)):
	maxval = np.amax(allcellsl9FH[row2])
	for value in range(np.size(allcellsl9FH,1)):	
		if allcellsl9FH[row2][value]>=maxval: 
			allcellsl9FHSpl[row2][value]=allcellsl9FH[row2][value]

for row2 in range(np.size(allcellsl9RH,0)):
	maxval = np.amax(allcellsl9RH[row2])
	for value in range(np.size(allcellsl9RH,1)):	
		if allcellsl9RH[row2][value]>=maxval: 
			allcellsl9RHSpl[row2][value]=allcellsl9RH[row2][value]

BothSplines9 = allcellsl9RHSpl + allcellsl9FHSpl

ispl = max_y/2
MaxInt = np.amax(BothSplines9)

mspl = 0
while ispl >=  0: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed9[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines9[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed9[ispl][mspl] = BothSplines9[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl-1

mspl = 0
ispl = max_y/2
while ispl< max_y: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed9[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines9[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed9[ispl][mspl] = BothSplines9[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl+1

BothFin9 = BothSplines + BothSplines9


MakeImagetrMicro(BothFin9, "Cells_SplineFin0_ "+str(l8Micro) + " to " + str(l9Micro),x0_c, y0_c)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin0_"+str(l8Micro) + "_" + str(l9Micro))


for row2 in range(np.size(BothFin9,0)): 
	for value in range(np.size(BothFin9,1)): 
		if BothFin9[row2][value]<MaxInt/2: 
			BothFin9[row2][value] = 0
print "Red"
MakeImagetrMicro(BothSplinesRed9, "Cells_Spline_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline_"+str(l8Micro) + "_" + str(l9Micro))

MakeImagetrMicro(BothSplines, "Cells_Spline0_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline0_"+str(l8Micro) + "_" + str(l9Micro))

MakeImagetrMicro(BothSplines9, "Cells_Spline01_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline01_"+str(l8Micro) + "_" + str(l9Micro))

MakeImagetrMicro(BothFin9, "Cells_SplineFin1_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin1_"+str(l8Micro) + "_" + str(l9Micro))

for row2 in range(np.size(BothFin9,0)): 
	for value in range(np.size(BothFin9,1)): 
		if BothFin9[row2][value]<MaxInt*0.6: 
			BothFin9[row2][value] = 0

MakeImagetrMicro(BothFin9, "Cells_SplineFin2_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin2_"+str(l8Micro) + "_" + str(l9Micro))

for row2 in range(np.size(BothFin9,0)): 
	for value in range(np.size(BothFin9,1)): 
		if BothFin9[row2][value]<MaxInt*0.65: 
			BothFin9[row2][value] = 0

MakeImagetrMicro(BothFin9, "Cells_SplineFin3_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin3_"+str(l8Micro) + "_" + str(l9Micro))

for row2 in range(np.size(BothFin9,0)): 
	for value in range(np.size(BothFin9,1)): 
		if BothFin9[row2][value]<MaxInt*0.7: 
			BothFin9[row2][value] = 0

MakeImagetrMicro(BothFin9, "Cells_SplineFin4_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin4_"+str(l8Micro) + "_" + str(l9Micro))

for row2 in range(np.size(BothFin9,0)): 
	for value in range(np.size(BothFin9,1)): 
		if BothFin9[row2][value]<MaxInt*0.75: 
			BothFin9[row2][value] = 0

MakeImagetrMicro(BothFin9, "Cells_SplineFin5_ "+str(l8Micro) + " to " + str(l9Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin5_"+str(l8Micro) + "_" + str(l9Micro))

###################l10#############################################

for ent in range(max_y/2): 
	for ent2 in range(max_x): 
		allcellsl10pos[ent][ent2] = allcellsl10[ent][ent2] 

allcellsl10posT = allcellsl10pos.transpose()


for row2 in range(np.size(allcellsl10posT,0)):
	maxval = np.amax(allcellsl10posT[row2])
#	print "row2: ", row2
#	print "maxval: ", maxval
	for value in range(np.size(allcellsl10posT,1)):	
		if allcellsl10posT[row2][value]>=maxval: 
#			print "yes if"
			allcellsl10posSpline[value][row2]=allcellsl10posT[row2][value]

for ent in range(max_y/2,max_y): 
	for ent2 in range(max_x): 
		allcellsl10neg[ent][ent2] = allcellsl10[ent][ent2] 

allcellsl10negT = allcellsl10neg.transpose()

for row2 in range(np.size(allcellsl10negT,0)):
	maxval = np.amax(allcellsl10negT[row2])
	for value in range(np.size(allcellsl10negT,1)):	
		if allcellsl10negT[row2][value]>=maxval: 
			allcellsl10negSpline[value][row2]=allcellsl10negT[row2][value]

BothSplines = allcellsl10posSpline + allcellsl10negSpline

for ent in range(max_y): 
	for ent2 in range(max_x/2): 
		allcellsl10FH[ent][ent2] = allcellsl10[ent][ent2] 

for ent in range(max_y): 
	for ent2 in range(max_x/2,max_x): 
		allcellsl10RH[ent][ent2] = allcellsl10[ent][ent2] 

for row2 in range(np.size(allcellsl10FH,0)):
	maxval = np.amax(allcellsl10FH[row2])
	for value in range(np.size(allcellsl10FH,1)):	
		if allcellsl10FH[row2][value]>=maxval: 
			allcellsl10FHSpl[row2][value]=allcellsl10FH[row2][value]

for row2 in range(np.size(allcellsl10RH,0)):
	maxval = np.amax(allcellsl10RH[row2])
	for value in range(np.size(allcellsl10RH,1)):	
		if allcellsl10RH[row2][value]>=maxval: 
			allcellsl10RHSpl[row2][value]=allcellsl10RH[row2][value]

BothSplines10 = allcellsl10RHSpl + allcellsl10FHSpl

ispl = max_y/2
MaxInt = np.amax(BothSplines10)

mspl = 0
while ispl >=  0: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed10[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines10[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed10[ispl][mspl] = BothSplines10[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl-1

mspl = 0
ispl = max_y/2
while ispl< max_y: 
	while mspl < max_x:

		if BothSplines[ispl][mspl] >= MaxInt/2: 
			BothSplinesRed10[ispl][mspl] = BothSplines[ispl][mspl]
		elif BothSplines10[ispl][mspl] >= MaxInt/2: 
				BothSplinesRed10[ispl][mspl] = BothSplines10[ispl][mspl]
		mspl = mspl+1
	mspl = 0
	ispl = ispl+1

BothFin10 = BothSplines + BothSplines10


MakeImagetrMicro(BothFin10, "Cells_SplineFin0_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin0_"+str(l9Micro) + "_" + str(l10Micro))


for row2 in range(np.size(BothFin10,0)): 
	for value in range(np.size(BothFin10,1)): 
		if BothFin10[row2][value]<MaxInt/2: 
			BothFin10[row2][value] = 0

print "Red"
MakeImagetrMicro(BothSplinesRed10, "Cells_Spline_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline_"+str(l9Micro) + "_" + str(l10Micro))

MakeImagetrMicro(BothSplines, "Cells_Spline0_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline0_"+str(l9Micro) + "_" + str(l10Micro))

MakeImagetrMicro(BothSplines10, "Cells_Spline01_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_Spline01_"+str(l9Micro) + "_" + str(l10Micro))

MakeImagetrMicro(BothFin10, "Cells_SplineFin1_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin1_"+str(l9Micro) + "_" + str(l10Micro))

for row2 in range(np.size(BothFin10,0)): 
	for value in range(np.size(BothFin10,1)): 
		if BothFin10[row2][value]<MaxInt*0.6: 
			BothFin10[row2][value] = 0

MakeImagetrMicro(BothFin10, "Cells_SplineFin2_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin2_"+str(l9Micro) + "_" + str(l10Micro))

for row2 in range(np.size(BothFin10,0)): 
	for value in range(np.size(BothFin10,1)): 
		if BothFin10[row2][value]<MaxInt*0.65: 
			BothFin10[row2][value] = 0

MakeImagetrMicro(BothFin10, "Cells_SplineFin3_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin3_"+str(l9Micro) + "_" + str(l10Micro))

for row2 in range(np.size(BothFin10,0)): 
	for value in range(np.size(BothFin10,1)): 
		if BothFin10[row2][value]<MaxInt*0.6: 
			BothFin10[row2][value] = 0

MakeImagetrMicro(BothFin10, "Cells_SplineFin4_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin4_"+str(l9Micro) + "_" + str(l10Micro))

for row2 in range(np.size(BothFin10,0)): 
	for value in range(np.size(BothFin10,1)): 
		if BothFin10[row2][value]<MaxInt*0.75: 
			BothFin10[row2][value] = 0

MakeImagetrMicro(BothFin10, "Cells_SplineFin5_ "+str(l9Micro) + " to " + str(l10Micro),x0_c-1, y0_c-1)
ShowImage(ImageLocOut +"/" + "_Cells_SplineFin5_"+str(l9Micro) + "_" + str(l10Micro))

#################

for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount





for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount
for row in range(np.size(allcells1,0)):
	for color in range(np.size(allcells1,1)): 
		allcells1Normed[row][color] = allcells1[row][color]/icount

MakeImage(allcellsNormed1,'Normalized')
ShowImage(ImageLoc+ Timepoint+"_Cells_Normalized")
	
