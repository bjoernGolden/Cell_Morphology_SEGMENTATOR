###Marie Hemmen, 26.04.16###

import sys
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


np.set_printoptions(threshold=np.nan)


def makeEllipse1(x0,y0,a,b,an):

	points=1000 #Number of points whicnh needs to construct the elipse
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
	plt.imshow(array,cmap = "YlGnBu") #YlGnBu, frueher: jet
	plt.colorbar()
	plt.title(titel)

def MakeImagetr(array,title,x0t,y0t):
	print "x0av: ", x0t, "y0av: ", y0t
	h, w = array.shape
	ax = plt.gca()

	plt.imshow(array,cmap = "YlGnBu",
			  extent=[-x0t, w-x0t, h-y0t, -y0t]) #YlGnBu
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('Number of Cells')
	ax.xaxis.set_label_coords(0.89, 0.45)
	ax.yaxis.set_label_coords(0.27, 0.93)

	plt.xlabel('Width [Pixel]')
	plt.ylabel('Length [Pixel]').set_rotation(0)
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
	plt.axis((-40,60 ,-40,40))

	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_fontsize(10)

	plt.title(title)


def MakeImagetrMicro(array,title,x0t2,y0t2):
	print "x0av: ", x0t2, "y0av: ", y0t2
	h, w = array.shape
	ax = plt.gca()

	plt.imshow(array,cmap = "YlGnBu",
			  extent=[-x0t2*0.13, (w-x0t2)*0.13, (h-y0t2)*0.13, -y0t2*0.13]) #YlGnBu
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('number of Cells')
	ax.xaxis.set_label_coords(0.89, 0.45)
	ax.yaxis.set_label_coords(0.3, 0.93)

	plt.ylabel('width [$\mu$m]').set_rotation(0)
	plt.xlabel('length [$\mu$m]')
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
		label.set_fontsize(15)

	plt.title(title)

def MakeNormedImage(array,title,x0t3,y0t3):
	h, w = array.shape
	ax = plt.gca()
	plt.imshow(array,cmap = "YlGnBu",vmin=0, vmax = 0.4,
				extent=[-x0t3*0.13, (w-x0t3)*0.13, (h-y0t3)*0.13, -y0t3*0.13])
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('normalized frequency of cell outlines', fontsize = 16)
	ax.xaxis.set_label_coords(0.8, 0.45)
	ax.yaxis.set_label_coords(0.21, 0.93)

	plt.ylabel('width [$\mu$m]', fontsize = 16).set_rotation(0)
	plt.xlabel('length [$\mu$m]', fontsize = 16)
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
		label.set_fontsize(15)

	plt.title(title)

def MakeNormalizedImage(array,title,x0t3,y0t3):
	h, w = array.shape
	ax = plt.gca()
	plt.imshow(array,cmap = "YlGnBu",
				extent=[-x0t3*0.13, (w-x0t3)*0.13, (h-y0t3)*0.13, -y0t3*0.13])
	cbar = plt.colorbar()
	cbar.ax.set_ylabel('normalized frequency of cell outlines', fontsize = 16)
	ax.xaxis.set_label_coords(0.80, 0.43)
	ax.yaxis.set_label_coords(0.20, 0.87)

	plt.ylabel('width [$\mu$m]', fontsize = 22).set_rotation(0)
	plt.xlabel('length [$\mu$m]', fontsize = 22)
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
		label.set_fontsize(18)

	plt.title(title)

def MakeImage2(array1,array2): 
	plt.imshow(array1,cmap = "gray", alpha = 0.5)
	plt.imshow(array2,cmap = "gray", alpha = 0.5)

def ShowImage(path): 
	plt.draw()
	plt.savefig(path + ".png")
	plt.show()

#parameterfile = sys.argv[1] #/home/marie/Master/Ellipseparameters/20160804_EKY360_fixiert_180_2_test.txt
#imagefile = sys.argv[2] #/home/marie/Master/Outlinecoordinates/Positives/20160804_EKY360_fixiert_180_2_test.txt
#StandDevLoc = sys.argv[3] #/home/marie/Master/Average_Images_New/
#StandDevFile = sys.argv[4] #EKY360_0
#StandDevFilePos = sys.argv[5] #EKY360_0_pos

parameterfile = sys.argv[1] #/home/marie/Master/Ellipseparameters/20160804_EKY360_fixiert_180_2_test.txt
imagefile = sys.argv[2] #/home/marie/Master/Outlinecoordinates/Positives/20160804_EKY360_fixiert_180_2_test.txt
ImageLoc = sys.argv[3] #/home/marie/Master/Average_Images_New/EKY360/
Matlabfile = sys.argv[4] #/home/marie/Master/Average_Images_New/MatlabFiles/EKY360/
Timepoint = sys.argv[5] #0
valuesfile = sys.argv[6] #/home/marie/Master/Values/EKY360/160
Strain = sys.argv[7] #EKY360
MatlabfileNorm = sys.argv[8] #/home/marie/Master/Average_Images_New/MatlabFilesNorm/EKY360/


wf = open(valuesfile, 'w')

f = open(parameterfile,'r')
sf = open(imagefile,'r')
majaxis_averagelist = []
minaxis_averagelist = []
majaxis2_averagelist = []
minaxis2_averagelist = []
area_averagelist = []
perimeter_averagelist = []
length_averagelist = []
x0_averagelist = []
y0_averagelist = []
x02_averagelist = []
y02_averagelist = []

majaxisMikro_averagelist = []
minaxisMikro_averagelist = []
majaxis2Mikro_averagelist = []
minaxis2Mikro_averagelist = []
areaMikro_averagelist = []
perimeterMikro_averagelist = []
lengthMikro_averagelist = []

All_Ellipses=[]
ellipseDict = {}

count = 0
icount = 0

f.readline()
sf.readline()
sline = sf.readline()
max_x = 160
max_y = 140
allarray = np.zeros((max_y,max_x)) 
arrafilledboth = np.zeros((max_y,max_x))
allcells = np.zeros((max_y,max_x))
allcellsPos = np.zeros((max_y,max_x))
allcellsPosNormed = np.zeros((max_y,max_x))
allcellsPosSpline = np.zeros((max_y,max_x))
allcellsSplBothPos = np.zeros((max_y,max_x))
allcellsFHSpl = np.zeros((max_y,max_x))
allcellsSHSpl = np.zeros((max_y,max_x))
allcellsSplinenoch = np.zeros((max_y,max_x))
allcellsNormed = np.zeros((max_y,max_x))
allcellsReduced = np.zeros((max_y,max_x))
allarrayreduced = np.zeros((max_y,max_x))
allarraydoubleReduced = np.zeros((max_y,max_x))
allcellsFH = np.zeros((max_y,max_x))
allcellsSH = np.zeros((max_y,max_x))
pluw = 0
allcellsNormalized = np.zeros((max_y, max_x))


for line in f: 
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
	if a>b: 
		majaxis = a 		#a ist major axis, b ist minor axis
		minaxis = b		
	else: 
		majaxis = b
		minaxis = a 
		
	if a2>b2: 
		majaxis2 = a2 		#a2 ist major axis, b2 ist minor axis
		minaxis2 = b2 
	else: 
		majaxis2 = b2
		minaxis2 = a2 
	print a,b,majaxis,minaxis
		
	#if area < 800: 
	#	sf.readline()
	#	sline = sf.readline()
	#	continue

	majaxis_averagelist.append(majaxis)
	majaxis2_averagelist.append(majaxis2)
	minaxis_averagelist.append(minaxis)
	minaxis2_averagelist.append(minaxis2)
	length_averagelist.append(length)
	area_averagelist.append(area)
	perimeter_averagelist.append(perimeter)

	majaxisMikro = majaxis*0.13
	minaxisMikro = minaxis*0.13
	majaxis2Mikro = majaxis2*0.13
	minaxis2Mikro = minaxis2*0.13
	areaMikro = area * 0.13 * 0.13
	perimeterMikro = perimeter * 0.13
	lengthMikro = length * 0.13
	majaxisMikro_averagelist.append(majaxisMikro)
	minaxisMikro_averagelist.append(minaxisMikro)
	majaxis2Mikro_averagelist.append(majaxis2Mikro)
	minaxis2Mikro_averagelist.append(minaxis2Mikro)
	areaMikro_averagelist.append(areaMikro)
	perimeterMikro_averagelist.append(perimeterMikro)
	lengthMikro_averagelist.append(lengthMikro)


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
	#plt.show()

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

	#MakeImage(barray,"")
	#ShowImage()
	#plt.show()
	allarray = allarray + barray
	#print "len allarray: ", len(allarray)

	arra1filled = ndimage.binary_fill_holes(arra1).astype(int)

	arra2filled = ndimage.binary_fill_holes(arra2).astype(int)

	arrafilledboth = arra1filled + arra2filled
	

	for h in range(np.size(arrafilledboth,0)):
		for u in range(np.size(arrafilledboth,1)):
			if arrafilledboth[h][u] > 1: 
				barray[h][u] = 0

	allarrayreduced = allarrayreduced + barray

	A = []
 	matrix =sline
 	tuple_rx = re.compile("\(\s*(\d+),\s*(\d+)\)")
 	for match in tuple_rx.finditer(matrix): 
 		A.append((int(match.group(1)),int(match.group(2))))
 	leng = len(A)
 	print "leng: ", leng
 	c = []
 	d = []

 	for i in A: 
		c.append(i[0])
		d.append(i[1])

	A=(c,d)
 	while leng < 70: 
 		sf.readline()
		sline = sf.readline()
 		A = []
 		matrix =sline
 		tuple_rx = re.compile("\(\s*(\d+),\s*(\d+)\)")
 		for match in tuple_rx.finditer(matrix): 
 			A.append((int(match.group(1)),int(match.group(2))))
 		leng = len(A)
 		c = []
 		d = []

 		for i in A: 
			c.append(i[0])
			d.append(i[1])
		plus = plus +1
		print "plus"

		A=(c,d)
	
	#if len(A[0]) == 0: 
	#	continue

	A=np.array(A)
	imAr = makeArray(A,max_x,max_y)

	el1=makeEllipse1(x0,y0,a,b,0)
	el2=makeEllipse1(x02,y02,a2,b2,0)


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

	RotEl = rotate(el1,theta)
	RotEl2 = rotate(el2,theta)

 	RotA = rotate(A,theta)
 	RotAPos = rotate(A,theta)
 	
 	RotAmaxx = np.amax(RotA[0])
 	RotAmaxy = np.amax(RotA[1])
###########################################hier!!!!!!!!#########################
  	RotAFH = rotate(A,theta)
  	RotASH = rotate(A,theta)

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

 	
	
	if RotAminx < 0: 
		RotA[0] = RotA[0]-RotAminx
		RotAPos[0] = RotAPos[0]-RotAminx
 	if RotAminy < 0: 
 		RotA[1] = RotA[1]-RotAminy
 		RotAPos[1] = RotAPos[1]-RotAminy

	
 	imageArray = makeArray(RotA,max_x,max_y)
 	imageArrayPos = makeArray(RotAPos,max_x,max_y)

 	imageArrayFH = makeArray(RotAFH,max_x,max_y)
 	imageArraySH = makeArray(RotASH,max_x,max_y)


 	allcells = allcells + imageArray
 	allcellsPos = allcellsPos + imageArrayPos
 	allcellsFH = allcellsFH + imageArrayFH
 	allcellsSH = allcellsSH + imageArraySH


	icount += 1
	sf.readline()
	sline = sf.readline()
	#print "count: ",icount
	#if icount > 15: 
	#	break

print "icound: ", icount

MakeImagetr(allarray,"",x0_c-1,y0_c-1)
ShowImage(ImageLoc + Timepoint + "_Ellipses")

MakeImagetrMicro(allarray,"",x0_c-1,y0_c-1)
ShowImage(ImageLoc + Timepoint + "_Ellipses_Micro")

#MakeImage(allarrayreduced, "Ellipses reduced")
#ShowImage(ImageLoc + Timepoint + "_Ellipses_reduced")

Cellfile = open(Matlabfile + Timepoint + "_Cells.txt","w")
for cell in allcells: 
	cell2 = str(cell)[1:-1]
	#print cell
	#print cell2
	Cellfile.write(cell2+"\n")

maxv = np.amax(allcells)
print "maxv: ", maxv

for row in range(np.size(allcells,0)):
	for color in range(np.size(allcells,1)): 
		allcellsNormed[row][color] = allcells[row][color]/icount
	

for row in range(np.size(allcellsPos,0)):
	for color in range(np.size(allcellsPos,1)): 
		allcellsPosNormed[row][color] = allcellsPos[row][color]/icount

for row in range(np.size(allcells,0)):
	for color in range(np.size(allcells,1)): 
		allcellsNormalized[row][color] = allcells[row][color]/maxv
	
CellfileNorm = open(MatlabfileNorm + Timepoint + "_Cells.txt","w")
for cell in allcellsNormalized: 
	cell2 = str(cell)[1:-1]
	#print cell
	#print cell2
	CellfileNorm.write(cell2+"\n")


MakeNormedImage(allcellsNormed,'',x0_c-1,y0_c-1)
ShowImage(ImageLoc+ Timepoint+"_Cells_Normalized")

MakeNormedImage(allcellsNormed,'',x0_c-1,y0_c-1)
ShowImage(ImageLoc+ Timepoint+"_Cells_Normalized_Micro")

MakeNormalizedImage(allcellsNormalized,'',x0_c-1,y0_c-1 )
ShowImage("/home/marie/Master/Thesis/Average_Images/NormalizedImages/" + Strain + "_" + Timepoint)

#MakeNormedImage(allcellsPosNormed,'Normalized')
#ShowImage(ImageLoc+ Timepoint+"_Cells_positiveValues_Normalized")


#MakeImage(allcellsFH,'FH')
#ShowImage(ImageLoc+ Timepoint+"_Cells_FH")

#MakeImage(allcellsSH,'SH')
#ShowImage(ImageLoc+ Timepoint+"_Cells_SH")


#MakeNormedImage(allcellsNormed,'Normalized')
#ShowImage(ImageLoc+ Timepoint+"_Cells_Normalized_highscale")




for row in range(np.size(allcells,0)):
	for color in range (np.size(allcells,1)):
		if allcells[row][color] > 45: 
			allcellsReduced[row][color] = allcells[row][color]

for r in range(np.size(allarrayreduced,0)):
	for c in range (np.size(allarrayreduced,1)):
		if allcells[r][c] > 45: 
			allarraydoubleReduced[r][c] = allarrayreduced[r][c]

#MakeImage(allarraydoubleReduced, "Ellipses double reduced")
#ShowImage(ImageLoc + Timepoint + "_Ellipses_double_reduced")

MakeImagetr(allcells, "",x0_c-1,y0_c-1)
ShowImage(ImageLoc + Timepoint + "_Cells")

MakeImagetrMicro(allcells, "",x0_c-1,y0_c-1)
ShowImage(ImageLoc + Timepoint + "_CellsMicro")




#MakeImage(allcellsReduced, "Cells reduced")
#ShowImage()

#print "allcellsPos: ", allcellsPos
#MakeImage(allcellsPos,"Cells Positive")
#ShowImage(ImageLoc + Timepoint + "_Cells_positiveValues")


allcellsPosT = allcellsPos.transpose()
#print "T: ",allcellsPosT
#allcellsFHT = allcellsFH.transpose()

#for col2 in range(np.size(allcellsFHT,1)):
#	maxval = np.amax(allcellsFHT[col2])
#	print "row2: ", row2
#	print "maxval: ", maxval
#	for value in range(np.size(allcellsFHT,0)):	
#		if allcellsFHT[value][col2]>=maxval: 
#			print "yes if"
#			allcellsFHSpl[col2][value]=allcellsFHT[value][col2]


for row2 in range(np.size(allcellsFH,0)):
	maxval = np.amax(allcellsFH[row2])
#	print "row2: ", row2
#	print "maxval: ", maxval
	for value in range(np.size(allcellsFH,1)):	
		if allcellsFH[row2][value]>=maxval: 
#			print "yes if"
			allcellsFHSpl[row2][value]=allcellsFH[row2][value]



#MakeImage(allcellsFHSpl,"SplineFH")
#ShowImage(ImageLoc + Timepoint + "_FH_Spline")

for row2 in range(np.size(allcellsSH,0)):
	maxval = np.amax(allcellsSH[row2])
#	print "row2: ", row2
#	print "maxval: ", maxval
	for value in range(np.size(allcellsSH,1)):	
		if allcellsSH[row2][value]>=maxval: 
#			print "yes if"
			allcellsSHSpl[row2][value]=allcellsSH[row2][value]
################HIER!!!!!!!!!!!

allcellsSHSpl[x0_c][y0_c] = 0
#MakeImage(allcellsSHSpl,"SplineSH")
#ShowImage(ImageLoc + Timepoint + "_SH_Spline")

#plt.imshow(allcellsSHSpl,cmap = "YlGnBu")
#plt.colorbar()
#plt.show()

allcellsSplBoth = allcellsFHSpl + allcellsSHSpl

#MakeImage(allcellsSplBoth,"BothSplines")
#ShowImage(ImageLoc + Timepoint + "_BothSplines")

for i in range(np.size(A[0])): 
 	
 		if RotA[1][i] > 0:
 			RotAPos[1][i] = -RotAPos[1][i] 
		RotA[0][i] = RotA[0][i]+x0_c
		RotA[1][i] = RotA[1][i]+y0_c
		RotAPos[0][i] = RotAPos[0][i]+x0_c
		RotAPos[1][i] = RotAPos[1][i]+y0_c

weg = 0
print y0_c
print "size: ",np.size(allcellsSplBoth,0)
for spl in range(y0_c+2,np.size(allcellsSplBoth,0)):
	for splx in range (np.size(allcellsSplBoth,1)):
		print spl
		print spl-weg
		allcellsSplBothPos[spl-weg][splx] = allcellsSplBoth[spl-weg][splx]+allcellsSplBoth[spl][splx]
		weg = weg*2
		if weg > y0_c: 
			break


#MakeImage(allcellsSplBothPos,"BothPos")
#ShowImage(ImageLoc + Timepoint + "_BothSplinesPos")

for row2 in range(np.size(allcellsPosT,0)):
	maxval = np.amax(allcellsPosT[row2])
#	print "row2: ", row2
#	print "maxval: ", maxval
	for value in range(np.size(allcellsPosT,1)):	
		if allcellsPosT[row2][value]>=maxval: 
#			print "yes if"
			allcellsPosSpline[value][row2]=allcellsPosT[row2][value]



#for i in range(np.size(allcells[0])): #damit um mittelpunkt rotiert wird#
	
#		allcells[0][i] = A[0][i]-x0
#		A[1][i] = A[1][i]-y0
#allcellsT = allcells.transpose()

#for line in range(np.size(allcellsT,1)):
#	maxvalue = np.amax(allcellsT[line])
#	for val in range(np.size(allcellsT,0)):
#		if allcellsT[line][val]>=maxval: 
#			allcellsSplinenoch[val][line] = allcellsT[line][val]

#MakeImage(allcellsSplinenoch,"Spline anders")
#ShowImage()


#print "allPosSpline", allcellsPosSpline
#print maxval
#MakeImage(allcellsPosSpline,"Spline")
#ShowImage(ImageLoc + Timepoint + "_Spline")



Cellfilepos = open(Matlabfile + Timepoint+"_pos.txt","w")
for cellpos in allcellsPosSpline:
	cellpos2 = str(cellpos)[1:-1]
	Cellfilepos.write(cellpos2 + "\n")


majaxis_average = np.mean(majaxis_averagelist)
minaxis_average = np.mean(minaxis_averagelist)
majaxis2_average = np.mean(majaxis2_averagelist)
minaxis2_average = np.mean(minaxis2_averagelist)
area_average = np.mean(area_averagelist)
perimeter_average = np.mean(perimeter_averagelist)
length_average = np.mean(length_averagelist)
x0_average = np.mean(x0_averagelist)
y0_average = np.mean(y0_averagelist)
x02_average = np.mean(x02_averagelist)
y02_average = np.mean(y02_averagelist)

majaxis_variance = np.var(majaxis_averagelist)
minaxis_variance = np.var(minaxis_averagelist)
majaxis2_variance = np.var(majaxis2_averagelist)
minaxis2_variance = np.var(minaxis2_averagelist)
area_variance = np.var(area_averagelist)
perimeter_variance = np.var(perimeter_averagelist)
length_variance = np.var(length_averagelist)
x0_variance = np.var(x0_averagelist)
y0_variance = np.var(y0_averagelist)
x02_variance = np.var(x02_averagelist)
y02_variance = np.var(y02_averagelist)

majaxis_sigma = np.sqrt(majaxis_variance)
minaxis_sigma = np.sqrt(minaxis_variance)
majaxis2_sigma = np.sqrt(majaxis2_variance)
minaxis2_sigma = np.sqrt(minaxis2_variance)
area_sigma = np.sqrt(area_variance)
perimeter_sigma = np.sqrt(perimeter_variance)
length_sigma = np.sqrt(length_variance)
x0_sigma = np.sqrt(x0_variance)
y0_sigma = np.sqrt(y0_variance)
x02_sigma = np.sqrt(x02_variance)
y02_sigma = np.sqrt(y02_variance)

majaxis_standardfehler = majaxis_sigma/np.sqrt(icount)
minaxis_standardfehler = minaxis_sigma/np.sqrt(icount)
majaxis2_standardfehler = majaxis2_sigma/np.sqrt(icount)
minaxis2_standardfehler = minaxis2_sigma/np.sqrt(icount)
area_standardfehler = area_sigma/np.sqrt(icount)
perimeter_standardfehler = perimeter_sigma/np.sqrt(icount)
x0_standardfehler = x0_sigma/np.sqrt(icount)
y0_standardfehler = y0_sigma/np.sqrt(icount)
x02_standardfehler = x02_sigma/np.sqrt(icount)
y02_standardfehler = y02_sigma/np.sqrt(icount)
length_standardfehler = length_sigma/np.sqrt(icount)



#Mikrometer
###########################################################

majaxisMikro_average = np.mean(majaxisMikro_averagelist)
minaxisMikro_average = np.mean(minaxisMikro_averagelist)
majaxis2Mikro_average = np.mean(majaxis2Mikro_averagelist)
minaxis2Mikro_average = np.mean(minaxis2Mikro_averagelist)
areaMikro_average = np.mean(areaMikro_averagelist)
perimeterMikro_average = np.mean(perimeterMikro_averagelist)
lengthMikro_average = np.mean(lengthMikro_averagelist)

majaxisMikro_variance = np.var(majaxisMikro_averagelist)
minaxisMikro_variance = np.var(minaxisMikro_averagelist)
majaxis2Mikro_variance = np.var(majaxis2Mikro_averagelist)
minaxis2Mikro_variance = np.var(minaxis2Mikro_averagelist)
areaMikro_variance = np.var(areaMikro_averagelist)
perimeterMikro_variance = np.var(perimeterMikro_averagelist)
lengthMikro_variance = np.var(lengthMikro_averagelist)

majaxisMikro_sigma = np.sqrt(majaxisMikro_variance)
minaxisMikro_sigma = np.sqrt(minaxisMikro_variance)
majaxis2Mikro_sigma = np.sqrt(majaxis2Mikro_variance)
minaxis2Mikro_sigma = np.sqrt(minaxis2Mikro_variance)
areaMikro_sigma = np.sqrt(areaMikro_variance)
perimeterMikro_sigma = np.sqrt(perimeterMikro_variance)
lengthMikro_sigma = np.sqrt(lengthMikro_variance)

majaxisMikro_standardfehler = majaxisMikro_sigma/np.sqrt(icount)
minaxisMikro_standardfehler = minaxisMikro_sigma/np.sqrt(icount)
majaxis2Mikro_standardfehler = majaxis2Mikro_sigma/np.sqrt(icount)
minaxis2Mikro_standardfehler = minaxis2Mikro_sigma/np.sqrt(icount)
areaMikro_standardfehler = areaMikro_sigma/np.sqrt(icount)
perimeterMikro_standardfehler = perimeterMikro_sigma/np.sqrt(icount)
lengthMikro_standardfehler = lengthMikro_sigma/np.sqrt(icount)

print "PIXEL"

print "majaxis: ",majaxis_average.round(2),"(" ,majaxis_variance.round(2),", ", majaxis_sigma.round(2),")"
print "minaxis: ", minaxis_average.round(2),"(" , minaxis_variance.round(2),", ", minaxis_sigma.round(2),")"
print "majaxis2: ", majaxis2_average.round(2), "(" ,majaxis2_variance.round(2), ", ",majaxis2_sigma.round(2),")"
print "minaxis2: ", minaxis2_average.round(2),"(" , minaxis2_variance.round(2), ", ",minaxis2_sigma.round(2),")"
print "length: ", length_average.round(2), "(" ,length_variance.round(2),", ", length_sigma.round(2),")"
print "area cell: ", area_average.round(2), "(" ,area_variance.round(2),", ", area_sigma.round(2),")"
print "perimeter vcell: ", perimeter_average.round(2),"(",perimeter_variance.round(2),", ",perimeter_sigma.round(2),")"


print "MIKROMETER"

print "majaxis: ",  majaxisMikro_average.round(2),"(" , majaxisMikro_variance.round(2), ", ",majaxisMikro_sigma.round(2), ", ", majaxisMikro_standardfehler.round(2),")"
print "minaxis: ", minaxisMikro_average.round(2), "(" ,minaxisMikro_variance.round(2), ", ",minaxisMikro_sigma.round(2), ", ",minaxisMikro_standardfehler.round(2),")"
print "majaxis2: ", majaxis2Mikro_average.round(2), "(" ,majaxis2Mikro_variance.round(2), ", ",majaxis2Mikro_sigma.round(2), ", ",majaxis2Mikro_standardfehler.round(2),")"
print "minaxis2: ", minaxis2Mikro_average.round(2), "(" ,minaxis2Mikro_variance.round(2), ", ",minaxis2Mikro_sigma.round(2), ", ",minaxis2Mikro_standardfehler.round(2),")"
print "length: ",lengthMikro_average.round(2), "(" ,lengthMikro_variance.round(2), ", ",lengthMikro_sigma.round(2), ", ",lengthMikro_standardfehler.round(2),")"
print "area cell: ", areaMikro_average.round(2), "(" ,areaMikro_variance.round(2), ", ",areaMikro_sigma.round(2),",", areaMikro_standardfehler.round(2),")"
print "perimeter vcell: ", perimeterMikro_average.round(2),"(" ,perimeterMikro_variance.round(2), ", ",perimeterMikro_sigma.round(2), ", ",perimeterMikro_standardfehler.round(2),")"
print "Segments: ", icount

wf.write("Parameter: " + "Value" + "Variance" + "Sigma" + "Standard error"+ "\n")

wf.write("Area in micrometer: " + str(areaMikro_average.round(2)) +","+ str(areaMikro_variance.round(2))+ ", "+str(areaMikro_sigma.round(2))+ ", "+ str(areaMikro_standardfehler.round(2))+"\n")
wf.write("Arealist: " + str(areaMikro_averagelist)+"\n")

wf.write("Perimeter in micrometer: " + str(perimeterMikro_average.round(2)) +","+str(perimeterMikro_variance.round(2))+ ", "+str(perimeterMikro_sigma.round(2))+ ", "+ str(perimeterMikro_standardfehler.round(2))+ "\n")
wf.write("Perimeterlist: " + str(perimeterMikro_averagelist)+"\n")

wf.write("Length in micrometer: " + str(lengthMikro_average.round(2)) + ","+str(lengthMikro_variance.round(2))+ ", "+str(lengthMikro_sigma.round(2))+ ", "+ str(lengthMikro_standardfehler.round(2))+"\n")
wf.write("Lengthlist: " + str(lengthMikro_averagelist)+"\n")

wf.write("majaxis in micrometer: " + str(majaxisMikro_average.round(2)) + ","+str(majaxisMikro_variance.round(2))+ ", "+str(majaxisMikro_sigma.round(2))+ ", "+ str(majaxisMikro_standardfehler.round(2))+"\n")
wf.write("majaxis list: " + str(majaxisMikro_averagelist)+"\n")

wf.write("minaxis in micrometer: " + str( minaxisMikro_average.round(2)) +","+str(minaxisMikro_variance.round(2))+ ", "+str(minaxisMikro_sigma.round(2))+ ", "+ str(minaxisMikro_standardfehler.round(2))+ "\n")
wf.write("minaxis list: " + str(minaxisMikro_averagelist)+"\n")

wf.write("majaxis2 in micrometer: " + str(majaxis2Mikro_average.round(2)) +","+ str(majaxis2Mikro_variance.round(2))+ ", "+str(majaxis2Mikro_sigma.round(2))+ ", "+ str(majaxis2Mikro_standardfehler.round(2))+"\n")
wf.write("majaxis2 list: " + str(majaxis2Mikro_averagelist)+"\n")

wf.write("minaxis2 in micrometer: " + str(minaxis2Mikro_average.round(2)) +","+ str(minaxis2Mikro_variance.round(2))+ ", "+str(minaxis2Mikro_sigma.round(2))+ ", "+ str(minaxis2Mikro_standardfehler.round(2))+"\n")
wf.write("minaxis2 list: " + str(minaxis2Mikro_averagelist)+"\n")

wf.write("Amount: " + str(icount))