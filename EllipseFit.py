###Marie Hemmen###
###22.05.2016###

import numpy as np
#from PIL import Image
from matplotlib import pyplot as plt
import scipy
from scipy import misc
from scipy import ndimage
from scipy import pi,sin,cos
from numpy import linspace
import math
import pdb
#import Image
from random import randint
import re
import os, sys

np.set_printoptions(threshold=np.nan)

#aenderung score: anfangen mit startscore von segment, dann ellipsen, die kleiner sind als z.B. drittel des startscores rausschmeissen



def makeEllipse1(x0,y0,a,b,an):

	points=1000 #Number of points (in x direction, to construct the ellipse)
	cos_a=cos(an*pi/180.)
	sin_a=sin(an*pi/180.)
	the=linspace(0,2*pi,points)
	#General ellpse, x0, y0 is the origin of the ellipse in xy plane
	X=a*cos(the)*cos_a-sin_a*b*sin(the)+x0
	Y=a*cos(the)*sin_a+cos_a*b*sin(the)+y0
	
	x_values=np.array([X])
	pos_y_values=np.array([Y])

	array_ellipse = np.append(x_values,pos_y_values, axis = 0)
	return array_ellipse

def scaling(array,sx,sy):
	array[0,:] = array[0,:]*sy
	array[1,:] = array[1,:]*sx
	
	return array

def makeArray(array): 
	modelarray=np.zeros((max_y,max_x)) 
	for m in range(array[0].size):
		x=array[0][m]-1
		y=array[1][m]-1
		if x<0: 
			x = 0
		if y<0: 
			y = 0
		modelarray[y][x]=1
	return modelarray

def Score(imageArrayfilled, ellipseArrayfilled): #arrays elementweise multiplizieren, dann summe über resultierendes array bilden
	SumArray=np.zeros((max_y,max_x))

	for i in range(max_y): 
		for m in range(max_x):
			if imageArrayfilled[i][m] == 0 and ellipseArrayfilled[i][m] == 0: 
				SumArray[i][m] = 1
			elif imageArrayfilled[i][m] == 1 and ellipseArrayfilled[i][m] ==1:
				SumArray[i][m] = 1
			elif imageArrayfilled[i][m] == 0 and ellipseArrayfilled[i][m] ==1:
				SumArray[i][m] = -1
			elif imageArrayfilled[i][m] == 1 and ellipseArrayfilled[i][m] ==0:
				SumArray[i][m] = -1
	
	sumValue=np.sum(SumArray)
	summe=np.sum(sumValue)

	return summe

def Score2(imageArrayfilled, ellipseArrayfilled, ellipseArrayfilled2): 
	SumArray=np.zeros((max_y,max_x))

	for i in range(max_y): 
		for m in range(max_x):
			if imageArrayfilled[i][m] == 0 and ellipseArrayfilled[i][m] == 0: 
				SumArray[i][m] = 1
			elif imageArrayfilled[i][m] == 1 and ellipseArrayfilled[i][m] ==1:
				SumArray[i][m] = 1
			elif imageArrayfilled[i][m] == 0 and ellipseArrayfilled[i][m] ==1:
				SumArray[i][m] = -1
			elif imageArrayfilled[i][m] == 1 and ellipseArrayfilled[i][m] ==0:
				SumArray[i][m] = -1
			if ellipseArrayfilled[i][m] == 1 and ellipseArrayfilled2[i][m] ==1: 
				SumArray[i][m] = SumArray[i][m]+2

	sumValue=np.sum(SumArray)
	summe=np.sum(sumValue)

	return summe

def MakeImage(array, array2, imagearray): 
	plt.imshow(array, cmap = "gray",alpha = 0.7)
	plt.imshow(array2,cmap = "gray", alpha = 0.5)
	plt.imshow(imagearray,cmap = "gray", alpha = 0.5)

def MakeImage1(array):
	plt.imshow(array)

def MakeLastImage(array1,array2,array3,array4): 
	plt.imshow(array1, cmap = "Blues")
	plt.imshow(array2,cmap = "gray", alpha = 0.5)
	plt.imshow(array3,cmap = "gray", alpha = 0.5)
	plt.imshow(array4,cmap = "gray", alpha = 0.5)

def longestDistance(M): # array of ellipse

	vnew=0
	distancelist=[]
	#print M

	for xy1 in range(np.size(M,1)): 
		for xy2 in range(np.size(M,1)): 
			x1=M[0][xy1]
			y1=M[1][xy1]

			x2=M[0][xy2]
			y2=M[1][xy2]

			vold=vnew
			v=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

			distancelist.append(v)
			if v>vold: 
				vnew=v
				maxx1=x1
				maxy1=y1
				maxx2=x2
				maxy2=y2
				
	maxdistance=max(distancelist)
	return maxdistance

def ShowImage(number1,number2,ImFile): 

	plt.draw()
	plt.savefig(ImFile+"/"+ str(number1)+"_"+str(number2)+".png")
	#plt.show()


def CreateParameterFile(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10,n11,n12,n13,n14,n15,n16): 
	with open(n1,"a+") as fi:
    		fi.write(str(n2)+"\t"+str(n3)+"\t"+str(n4)+"\t"+str(n5)+"\t"+str(n6)+"\t"+str(n7)+"\t"+str(n8)+"\t"+str(n9)+"\t"+str(n10)+"\t"+str(n11)+"\t"+str(n12)+"\t"+str(n13)+"\t"+str(n14)+"\t"+str(n15)+"\t"+str(n16)+"\n")


#OutlinePathIn = sys.argv[1] #/home/marie/Master/Outlinecoordinates/Positives
#Outlines = os.listdir(OutlinePathIn)
#parameterfileOut = sys.argv[2] #/home/marie/Master/Ellipseparameter
#ImagesFileOut = sys.argv[3] #/home/marie/Master/EllipseFitImages

OutlinePathIn = sys.argv[1] #/home/marie/Master/Outlinecoordinates/PositivesNew
Outlinefile = sys.argv[2] #20160608_EKY360
parameterfileOut = sys.argv[3] #/home/marie/Master/EllipseparametersNew
ImagesFileOut = sys.argv[4] #/home/marie/Master/EllipseFitImagesNew

#for Outlinefile in Outlines:

Coordinates = OutlinePathIn + "/" + Outlinefile + ".txt" #name of matlab textfile with pixel coordinates of segment

Parameterfile = parameterfileOut + "/" + Outlinefile + ".txt"
if os.path.exists(Parameterfile):
	PFile = open(Parameterfile,"w")
	PFile.seek(0)
	CreateParameterFile(Parameterfile, "number", "a", "b", "a2", "b2", "area", "perimeter", "ratio", "length","x0","y0","x02","y02", "theta1", "theta2")

else:

	CreateParameterFile(Parameterfile, "number", "a", "b", "a2", "b2", "area", "perimeter", "ratio", "length","x0","y0","x02","y02","theta1", "theta2")

print "type: ", type(Outlinefile)
#Outlinefileshort = Outlinefile[:-4]
#print "Outlinefileshort:",Outlinefileshort
Imagefile = ImagesFileOut + "/" + Outlinefile 
number=1

#print "Files: ", Outlines
print "CoordinateFile: ", Coordinates 

f = open(Coordinates,'r')


for line in f: 
	print "number: ", number
	sumdic = {}
	sumdic2 = {}
	area=0
	areaReduced = 0
	perimeter = 0

	A = []
	matrix =line
	tuple_rx = re.compile("\(\s*(\d+),\s*(\d+)\)")
	for match in tuple_rx.finditer(matrix): 
		A.append((int(match.group(1)),int(match.group(2))))
	if len(A)<70: 
		continue
		
	max_x=0
	max_y=0

	for i in A: #!!!!!!!!!!!!!!!!umaendern
		if i[0]>max_x: #np.amax()
			max_x=i[0]
		if i[1]>max_y: 
			max_y=i[1]

	imageArray=np.zeros((max_y,max_x))
	fakeArray=np.zeros((max_y,max_x))

	for i in A: 
		x=i[0]-1
		y=i[1]-1
		imageArray[y][x]=1

	perimeter = np.sum(imageArray)

	imageArrayfilled=ndimage.binary_fill_holes(imageArray).astype(int)

	for i in range(100): 
		an=0
		a = randint(1,max_x/2)  #a = randint(5,max_x/2)
		b = randint(1,max_y/2)
		x0 = randint(a,max_x-a)
		y0 = randint(b,max_y-b)

		#def makeEllipse1(x0,y0,a,b), x0>a, y0>b, 2a<=max_x bzw. a<=max_x/2, a+x0<=max_x bzw. x0 <= max_x-a
		ellipse = makeEllipse1(x0,y0,a,b,an)
		ellipseArray = makeArray(ellipse)

		ellipseArrayfilled=ndimage.binary_fill_holes(ellipseArray).astype(int)
		
		score = Score(imageArrayfilled, ellipseArrayfilled)
		oldscore = score


		sumdic[score] = [x0,y0,a,b,an]
		x0old = x0

#transition of first ellipse in x direction
##################################################################################################################
		if x0>a+1:
			x0 = x0-1
			
			ellipse = makeEllipse1(x0,y0,a,b,an) 
			ellipseArray = makeArray(ellipse)
			ellipseArrayfilled=ndimage.binary_fill_holes(ellipseArray).astype(int)
		
			score = Score(imageArrayfilled, ellipseArrayfilled)

		if score<=oldscore and x0+2<(max_x-a): 
			x0 = x0+2

			ellipse = makeEllipse1(x0,y0,a,b,an)
			ellipseArray = makeArray(ellipse) 	#statt jedes mal neuen array lieber in array jedes pixel um eins nach rechts schieben
			ellipseArrayfilled=ndimage.binary_fill_holes(ellipseArray).astype(int)
		
			score = Score(imageArrayfilled, ellipseArrayfilled)

		while score >= oldscore and a<x0<(max_x-a): 
			ellipse = makeEllipse1(x0,y0,a,b,an)
			ellipseArray = makeArray(ellipse)
			ellipseArrayfilled=ndimage.binary_fill_holes(ellipseArray).astype(int)
				
			oldscore = score
			score = Score(imageArrayfilled, ellipseArrayfilled) 
			sumdic[score] = [x0,y0,a,b,an]

			if x0old > x0:
				x0old = x0 
				x0 = x0-1
			else: 
				x0old = x0 
				x0 = x0+1

		x0 = x0old

#transition of first ellipse in y direction
#################################################################################################################
		sumdic[score] = [x0,y0,a,b,an]
		y0old = y0

		if y0>b+1:
			y0 = y0-1
		
		ellipse = makeEllipse1(x0,y0,a,b,an)
		ellipseArray = makeArray(ellipse)
		ellipseArrayfilled=ndimage.binary_fill_holes(ellipseArray).astype(int)
		
		score = Score(imageArrayfilled, ellipseArrayfilled)

		if score<=oldscore and y0+2<(max_y-b): 
			y0 = y0+2

		ellipse = makeEllipse1(x0,y0,a,b,an)
		ellipseArray = makeArray(ellipse)
		ellipseArrayfilled=ndimage.binary_fill_holes(ellipseArray).astype(int)
		
		score = Score(imageArrayfilled, ellipseArrayfilled)

		while score >= oldscore and b<y0<(max_y-b): 
			ellipse = makeEllipse1(x0,y0,a,b,an)
			ellipseArray = makeArray(ellipse)
			ellipseArrayfilled=ndimage.binary_fill_holes(ellipseArray).astype(int)
				
			oldscore = score
			score = Score(imageArrayfilled, ellipseArrayfilled) 
			sumdic[score] = [x0,y0,a,b,an]

			if y0old > y0:
				y0old = y0 
				y0 = y0-1
			else: 
				y0old = y0 
				y0 = y0+1

		y0 = y0old

	SortedKeys = sorted(sumdic.keys())
	BestScore = SortedKeys[-1:][0]
	SecondBestScore = SortedKeys[-2:][0]

	maxsumParametersB = sumdic[BestScore]
	x0b,y0b,ab,bb,an = maxsumParametersB

	maxsumParametersS = sumdic[SecondBestScore]
	x0S,y0S,aS,bS,an = maxsumParametersS

#Rotation of Best Ellipse	
###################################################################################################################

	bestEllipse1 = makeEllipse1(x0b,y0b,ab,bb,an)
	BestEllipseArray = makeArray(bestEllipse1)
	BestEllipseArrayfilled = ndimage.binary_fill_holes(BestEllipseArray).astype(int)
	#MakeImage(BestEllipseArray, fakeArray, imageArrayfilled)
	#plt.show()
	an = -50
	while an < 50:  
		Y = makeEllipse1(x0b,y0b,ab,bb,an)

		if np.amax(Y[0])<max_x and np.amax(Y[1])<max_y:

			Ynew=makeArray(Y)
			Ynewfilled=ndimage.binary_fill_holes(Ynew).astype(int)
			
			#MakeImage(BestEllipseArray, Ynew, imageArrayfilled)
			#plt.show()

			score = Score(imageArrayfilled, Ynewfilled) #wenn score kleiner wird brich ab!

			sumdic[score] = [x0b,y0b,ab,bb,an]
		an = an+5

#Rotation of second best ellipse
#################################################################################################################
	
	an = 0
	bestEllipse1S = makeEllipse1(x0S,y0S,aS,bS,an)
	BestEllipseArrayS = makeArray(bestEllipse1S)
	BestEllipseArrayfilledS = ndimage.binary_fill_holes(BestEllipseArrayS).astype(int)
	#MakeImage(BestEllipseArrayS, fakeArray, imageArrayfilled)
	#plt.show()
	an = -50
	while an < 50:  
		Y = makeEllipse1(x0b,y0b,ab,bb,an)

		if np.amax(Y[0])<max_x and np.amax(Y[1])<max_y:

			Ynew=makeArray(Y)
			Ynewfilled=ndimage.binary_fill_holes(Ynew).astype(int)
			
			#MakeImage(BestEllipseArrayS, Ynew, imageArrayfilled)
			#plt.show()

			score = Score(imageArrayfilled, Ynewfilled) 

			sumdic[score] = [x0b,y0b,ab,bb,an]
		an = an+5

#very best ellipse
##################################################################################################################
	Best = np.amax(sumdic.keys())

	BestParameters = sumdic[Best]
	#print "sumdic: ",sumdic[Best]

	x0,y0,a,b,an = BestParameters
	theta1 = an

	BestEllipse = makeEllipse1(x0,y0,a,b,an)

	BestEllipseArray = makeArray(BestEllipse)
	BestEllipseArrayfilled = ndimage.binary_fill_holes(BestEllipseArray).astype(int)
	MakeImage(BestEllipseArrayfilled,fakeArray,imageArray)
	#plt.show()
	oldfilled = np.array(imageArrayfilled)

#reduce filled cell in order to find the small ellipse (shmoo)
##################################################################################################################
	area = np.sum(imageArrayfilled)
	ratio = float(area)/float(perimeter)


	for entry in range(np.size(imageArrayfilled,0)): #nur arrays voneinander abziehen, statt durch alle indizes gehen!
		for value in range(np.size(imageArrayfilled,1)):
			imageArrayfilled[entry][value] = imageArrayfilled[entry][value] - BestEllipseArrayfilled[entry][value]
			if imageArrayfilled[entry][value] < 0: 
				imageArrayfilled[entry][value] = 0

#check if cell has a shmoo, if not, don't fit a second ellipse and continue with next shmoo image!
################################################################################################################
	areaReduced = np.sum(imageArrayfilled)
	#print "area: ", area

	#print "areareduce: ", areaReduced

	#print 1./7.*float(area)
	# if float(areaReduced) < 1./8.*float(area): 
	# 	print "huuuuuuuuuuuuuu number"
	# 	length = longestDistance(BestEllipse)
	# 	CreateParameterFile(Parameterfile, number, a, b, 0, 0, area, perimeter, ratio, length,x0,y0,0,0, theta1, 0)
	# 	MakeImage(imageArray,BestEllipseArray,fakeArray)
	# 	ShowImage(number,1,Imagefile)

	# 	number=number+1
	# 	continue

#second ellipse
####################################################################################################################

	for i in range(350): #350
		an = 0
		a2 = randint(1,max_x/4) 	#parameter einschraenken, weil man weiß wieviele pixel in shmoo sind!
		b2 = randint(1,max_y/4)
		x02 = randint(a2,max_x-a2)
		y02 = randint(b2,max_y-b2)

		#def makeEllipse1(x0,y0,a,b), x0>a, y0>b, 2a<=max_x bzw. a<=max_x/2, a+x0<=max_x bzw. x0 <= max_x-a
		ellipse2 = makeEllipse1(x02,y02,a2,b2,an)
		ellipseArray2 = makeArray(ellipse2)

		ellipseArrayfilled2=ndimage.binary_fill_holes(ellipseArray2).astype(int)

		score2 = Score2(imageArrayfilled, ellipseArrayfilled2,BestEllipseArrayfilled)
		oldscore2 = score2

		sumdic2[score2] = [x02,y02,a2,b2,an]
		x0old2 = x02

#transition of second ellipse in x direction
##########################################################################################################

		if x02>a2+1:
			x02 = x02-1

		ellipse2 = makeEllipse1(x02,y02,a2,b2,an)
		ellipseArray2 = makeArray(ellipse2)
		ellipseArrayfilled2=ndimage.binary_fill_holes(ellipseArray2).astype(int)
		
		score2 = Score2(imageArrayfilled, ellipseArrayfilled2,BestEllipseArrayfilled)

		if score2<=oldscore2 and x02+2<(max_x-a2): 
			x02 = x02+2

		ellipse2 = makeEllipse1(x02,y02,a2,b2,an)
		ellipseArray2 = makeArray(ellipse2)
		ellipseArrayfilled2=ndimage.binary_fill_holes(ellipseArray2).astype(int)
		
		score2 = Score2(imageArrayfilled, ellipseArrayfilled2,BestEllipseArrayfilled)

		while score2 >= oldscore2 and a2<x02<(max_x-a2): 
			ellipse2 = makeEllipse1(x02,y02,a2,b2,an)
			ellipseArray2 = makeArray(ellipse2)
			ellipseArrayfilled2=ndimage.binary_fill_holes(ellipseArray2).astype(int)
				
			oldscore2 = score2
			score2 = Score2(imageArrayfilled, ellipseArrayfilled2,BestEllipseArrayfilled) 
			sumdic2[score2] = [x02,y02,a2,b2,an]

			if x0old2 > x02:
				x0old2 = x02 
				x02 = x02-1
			else: 
				x0old2 = x02 
				x02 = x02+1

		x02 = x0old2

#transition of second ellipse in y direction
###################################################################################################
		
		sumdic2[score2] = [x02,y02,a2,b2,an]
		y0old2 = y02

		if y02>b2+1:
			y02 = y02-1
	
		ellipse2 = makeEllipse1(x02,y02,a2,b2,an)
		ellipseArray2 = makeArray(ellipse2)
		ellipseArrayfilled2=ndimage.binary_fill_holes(ellipseArray2).astype(int)
		
		score2 = Score2(imageArrayfilled, ellipseArrayfilled2,BestEllipseArrayfilled)

		if score2<=oldscore2 and y02+2<(max_y-b2): 
			y02 = y02+2

		ellipse2 = makeEllipse1(x02,y02,a2,b2,an)
		ellipseArray2 = makeArray(ellipse2)
		ellipseArrayfilled2=ndimage.binary_fill_holes(ellipseArray2).astype(int)
		
		score2 = Score2(imageArrayfilled, ellipseArrayfilled2,BestEllipseArrayfilled)

		while score2 >= oldscore2 and b2<y02<(max_y-b2): 
			ellipse2 = makeEllipse1(x02,y02,a2,b2,an)
			ellipseArray2 = makeArray(ellipse2)
			ellipseArrayfilled2=ndimage.binary_fill_holes(ellipseArray2).astype(int)
				
			oldscore2 = score2
			score2 = Score2(imageArrayfilled, ellipseArrayfilled2,BestEllipseArrayfilled) 
			sumdic2[score2] = [x02,y02,a2,b2,an]

			if y0old2 > y02:
				y0old2 = y02 
				y02 = y02-1
			else: 
				y0old2 = y02 
				y02 = y02+1

		y02 = y0old2

	SortedKeys2 = sorted(sumdic2.keys())
	BestScore2 = SortedKeys2[-1:][0]
	SecondBestScore2 = SortedKeys2[-2:][0]

	maxsumParametersB2 = sumdic2[BestScore2]
	x0b2,y0b2,ab2,bb2,an = maxsumParametersB2

	maxsumParametersS2 = sumdic2[SecondBestScore2]
	x0S2,y0S2,aS2,bS2,an = maxsumParametersS2

	
	
#Rotation of second Best Ellipse
###################################################################################################################

	an = 0
	bestEllipse2 = makeEllipse1(x0b2,y0b2,ab2,bb2,an)
	BestEllipseArray2 = makeArray(bestEllipse2)
	BestEllipseArrayfilled2 = ndimage.binary_fill_holes(BestEllipseArray2).astype(int)
	an = -30
	while an < 30:  
		Y = makeEllipse1(x0b2,y0b2,ab2,bb2,an)

		if np.amax(Y[0])<max_x and np.amax(Y[1])<max_y:

			Ynew=makeArray(Y)
			Ynewfilled=ndimage.binary_fill_holes(Ynew).astype(int)
			
			#MakeImage(BestEllipseArray2, Ynew, imageArrayfilled)
			#plt.show()

			score = Score2(imageArrayfilled, Ynewfilled,BestEllipseArrayfilled) 

			sumdic2[score] = [x0b2,y0b2,ab2,bb2,an]
		an = an+2

#Rotation of second best ellipse
#################################################################################################################

	an = 0
	bestEllipse2S = makeEllipse1(x0S2,y0S2,aS2,bS2,an)
	BestEllipseArrayS2 = makeArray(bestEllipse2S)
	BestEllipseArrayfilledS2 = ndimage.binary_fill_holes(BestEllipseArrayS2).astype(int)
	an = -30
	while an < 30:  
		Y = makeEllipse1(x0S2,y0S2,aS2,bS2,an)

		if np.amax(Y[0])<max_x and np.amax(Y[1])<max_y:

			Ynew=makeArray(Y)
			Ynewfilled=ndimage.binary_fill_holes(Ynew).astype(int)
			
			#MakeImage(BestEllipseArray2S, Ynew, imageArrayfilled)
			#plt.show()

			score = Score2(imageArrayfilled, Ynewfilled,BestEllipseArrayfilled) 

			sumdic2[score] = [x0S2,y0S2,aS2,bS2,an]
		an = an+2	

#very best ellipse
##################################################################################################################
	Best2 = np.amax(sumdic2.keys())

	BestParameters2 = sumdic2[Best2]
	intersectionlist=[]

	#print "len5"
	x02,y02,a2,b2,an2 = BestParameters2
	theta2 = an2
	#print sumdic2, Best2,BestParameters2,"max_x: ", max_x,"max_y: ", max_y
	#pdb.set_trace()
	BestEllipse2 = makeEllipse1(x02,y02,a2,b2,an2)  #DAS HIER WURDE GEAENDERT!!!

	#print "best Ellipse 2: ", BestEllipse2
	#hier gibts manchmal noch index out of bounds fehler!!!!!!!!
	############################
	BestEllipseArray2 = makeArray(BestEllipse2)
	BestEllipseArrayfilled2 = ndimage.binary_fill_holes(BestEllipseArray2).astype(int)

	#center of ellipse1 and 2
	###########################
	m1x = float(x0)
	m1y=float(y0)
	m2x=float(x02)
	m2y=float(y02)

	middleArray1=np.zeros((max_y,max_x))
	middleArray2=np.zeros((max_y,max_x))

	middleArray1[m1y][m1x]=1
	middleArray2[m2y][m2x]=1

	#line through ellipse centers
	###############################

	if m2x == m1x: 
		print "len5if"
		xline = np.linspace(m1x,m1x,100)		#array mit 2000 eintraegen, jeder hat den x wert des mittelpunkts der beiden ellipsen
		yline = np.linspace(0,max_y-1,100)	# array mit 2000 eingraegen, von 0 bis maxy-1
		x_array=np.array([xline])
		y_array=np.array([yline])

		array_line = np.append(x_array,y_array, axis = 0)
		linearray=np.zeros((max_y,max_x)) #hier unbedingt noch aendern!!!!! problem beim naechsten schritt: values von ellipse koennen hoeher als max_x und max_y des images sein!!!!!!!
		for m in range(array_line[0].size):
			xval=array_line[0][m]
			yval=array_line[1][m]
			linearray[yval][xval]=1
		

	else:

		print "len5else", max_x,m1x,m1y
		m = (m2y-m1y)/(m2x-m1x)					#steigung
		xline=np.linspace(0,max_x-1, 10000)		#

		yline = np.around(m*(xline-m1x)+m1y)	#geradengleichung: y = m*x

		x_array=np.array([xline])
		y_array=np.array([yline])

		array_line = np.append(x_array,y_array, axis = 0)
	
		linearray=np.zeros((max_y,max_x)) #hier unbedingt noch aendern!!!!! problem beim naechsten schritt: values von ellipse koennen hoeher als max_x und max_y des images sein!!!!!!!
		for m in range(array_line[0].size):
			xval=array_line[0][m]
			yval=array_line[1][m]
			if yval >= max_y or yval < 0: 
				continue
			linearray[yval][xval]=1

	
			
	intersectionarray = linearray * imageArray
	
	valuesx = []
	valuesy = []
	for xvalue in range(np.size(intersectionarray,1)): 
		for yvalue in range(np.size(intersectionarray,0)):
			if intersectionarray[yvalue][xvalue] == 1: 
				valuesx.append(xvalue)
				valuesy.append(yvalue)
	print valuesx, valuesy
	if len(valuesx)<2: 
		print "con"
		continue

	if len(valuesy)<2: 
		print "cony"
		continue
	xlen = len(valuesx)
	ylen = len(valuesy)

	if xlen != ylen: 		#just in case, um programmabsturz zu vermeiden
		continue

	x1v = valuesx[0]
	x2v = valuesx[xlen-1]	#ueberschneidung von den 2 arrays kann mehr als zwei punkte sein (z.B. linie an einem ende), erster und letzter eintrag sind an verschiedenen enden des shmooarrays

	y1v = valuesy[0]
	y2v = valuesy[ylen-1]

	length = np.sqrt((x2v-x1v)*(x2v-x1v)+(y2v-y1v)*(y2v-y1v))

	

	#length = np.sum(intersectionarray) 
	print "length: ", length
	CreateParameterFile(Parameterfile, number, a, b, a2, b2, area, perimeter, ratio, length,x0,y0,x02,y02, theta1, theta2)	#mehr infos in dieses file, z.B.gentyp
		
	MakeImage(imageArray,BestEllipseArray,BestEllipseArray2)
	ShowImage(number,1,Imagefile)

	MakeImage(imageArray,intersectionarray, fakeArray)
	ShowImage(number,2,Imagefile)

	#MakeLastImage(imageArray,intersectionarray, BestEllipseArray,BestEllipseArray2)
	#ShowImage(number,3)

	number=number+1

	









