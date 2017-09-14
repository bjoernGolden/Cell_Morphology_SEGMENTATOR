import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class Spline_Interpolation:
    ''' Interpolate natural cell shape with spline
        and obtain curvatures '''

    pointsShape = []
    u = []
    tck = []
    spline = []
    Dspline_Du = []
    D2spline_D2u = []
    kappa_m = []
    kappa_theta = []
    total_arclength = 0.0

    def __init__(self, points, start_point, step_size = 0.005, smoothing = 0.01,  show=True):

        self.pointsShape = []
        self.u = []
        self.tck = []
        self.splineShape = []
        self.splineShape_upHalf = []
        self.splineShape_s = []
        self.Dspline_Du = []
        self.D2spline_D2u = []
        self.Dspline_Du_s = []
        self.D2spline_D2u_s = []
        self.start_index_spline = 0
        self.kappa_m = []
        self.kappa_theta = []

        self.interpolate(points, step_size, smoothing)

        self.construct_arclength_spline(start_point, True)

        self.calculate_meridionalCurvature(True)
        self.calculate_circumferentialCurvature(True)



    def interpolate(self, points, step_size, smoothing):

        for pt in points:
            self.pointsShape.append(pt)

            # temp = len(points) - 1
            # for i in range(0, len(self.pointsShape)):
            # x = self.pointsShape[temp - i][0]
            # y = -self.pointsShape[temp - i][1]
            # self.pointsShape.append([x, y])

        # self.pointsShape.append([0, 0.0])
        # self.pointsShape.insert(0, [0, 0.0])

        self.pointsShape = np.array(self.pointsShape)
        x = self.pointsShape[:, 0]
        y = self.pointsShape[:, 1]

        # self.tck, self.u = interpolate.splprep([x, y], k=5, s=0.1)
        self.tck, self.u = interpolate.splprep([x, y], k=5, s=0.01)
        # unew = np.arange(0,1.01,0.01)
        self.u = np.arange(0, 1.01, step_size)
        self.splineShape = interpolate.splev(self.u, self.tck)

        # ''' Calcualte Derivatives and upper half'''

        Dspline_Du_temp = interpolate.splev(self.u, self.tck, der=1)
        D2spline_D2u_temp = interpolate.splev(self.u, self.tck, der=2)
        
        for i in range(0, len(self.splineShape[0])):
            if (self.splineShape[1][i] > 0):
                self.splineShape_upHalf.append([self.splineShape[0][i], self.splineShape[1][i]])
                self.Dspline_Du.append([Dspline_Du_temp[0][i], Dspline_Du_temp[1][i]])
                self.D2spline_D2u.append([D2spline_D2u_temp[0][i], D2spline_D2u_temp[1][i]])
        
        self.total_arclength = 0.0
	for i in range(0,len(self.splineShape_upHalf)-1):
	  temp = np.sqrt((self.splineShape_upHalf[i][0] - self.splineShape_upHalf[i + 1][0]) ** 2 + (
                    self.splineShape_upHalf[i][1] - self.splineShape_upHalf[i + 1][1]) ** 2)
	  self.total_arclength += temp
                
                
    def arclength_to_x(self,s):
	
	x= 0.0
	arc_s = 0.0
	
	for i in range(0,len(self.splineShape_upHalf)-1):
	  temp = np.sqrt((self.splineShape_upHalf[i][0] - self.splineShape_upHalf[i + 1][0]) ** 2 + (
                    self.splineShape_upHalf[i][1] - self.splineShape_upHalf[i + 1][1]) ** 2)
	  arc_s += temp
	  if (arc_s >= s):
	    x = self.splineShape_upHalf[i][0]
	    break
	  
	return x

    def construct_arclength_spline(self, start_point, reverse = False):

        start_index = 0
        min_distance = np.sqrt((start_point[0] - self.splineShape_upHalf[0][0]) ** 2 + (
            start_point[1] - self.splineShape_upHalf[0][1]) ** 2)

        for i in range(0, len(self.splineShape_upHalf)):
            temp = np.sqrt((start_point[0] - self.splineShape_upHalf[i][0]) ** 2 + (
                start_point[1] - self.splineShape_upHalf[i][1]) ** 2)
            if (temp < min_distance):
                start_index = i
                min_distance = temp

        arclength = 0.0

        if (not reverse):
            for i in range(start_index, len(self.splineShape_upHalf) - 1):
                ds = np.sqrt((self.splineShape_upHalf[i][0] - self.splineShape_upHalf[i + 1][0]) ** 2 + (
                    self.splineShape_upHalf[i][1] - self.splineShape_upHalf[i + 1][1]) ** 2)
                self.splineShape_s.append([self.splineShape_upHalf[i][0], self.splineShape_upHalf[i][1], arclength])
                self.Dspline_Du_s.append([self.Dspline_Du[i][0], self.Dspline_Du[i][1], arclength])
                self.D2spline_D2u_s.append([self.D2spline_D2u[i][0], self.D2spline_D2u[i][1], arclength])
                arclength += ds

            self.Dspline_Du_s.append([self.Dspline_Du[-1][0], self.Dspline_Du[-1][1], arclength])
            self.D2spline_D2u_s.append([self.D2spline_D2u[-1][0], self.D2spline_D2u[-1][1], arclength])
            self.splineShape_s.append([self.splineShape_upHalf[-1][0], self.splineShape_upHalf[-1][1], arclength])

        else:
            num_points = len(self.splineShape_upHalf)
            for i in range(0, start_index-1):
                ds = np.sqrt((self.splineShape_upHalf[start_index - i][0] - self.splineShape_upHalf[start_index - i - 1][0]) ** 2 + (
                    self.splineShape_upHalf[start_index - i][1] - self.splineShape_upHalf[start_index - i - 1][1]) ** 2)
                self.splineShape_s.append([self.splineShape_upHalf[start_index - i][0], self.splineShape_upHalf[start_index - i][1], arclength])
                self.Dspline_Du_s.append([self.Dspline_Du[start_index - i][0], self.Dspline_Du[start_index - i][1], arclength])
                self.D2spline_D2u_s.append([self.D2spline_D2u[start_index - i][0], self.D2spline_D2u[start_index - i][1], arclength])
                arclength += ds

            self.Dspline_Du_s.append(
                [self.Dspline_Du[0][0], self.Dspline_Du[0][1], arclength])
            self.D2spline_D2u_s.append(
                [self.D2spline_D2u[0][0], self.D2spline_D2u[0][1], arclength])
            self.splineShape_s.append([self.splineShape_upHalf[0][0], self.splineShape_upHalf[0][1], arclength])





    def show_naturalshape(self):
        ''' Plot the spline '''

        x = self.pointsShape[:, 0]
        y = self.pointsShape[:, 1]

        #plt.figure()
        #plt.plot(x, y, 'x', self.splineShape[0], self.splineShape[1], 'b')
        #out = np.array(self.splineShape_s)
        #x = out[:, 0]
        #y = out[:, 1]
        #plt.plot(x, y, color = 'r')
        #plt.title('Spline of parametrically-defined curve')
        #plt.show()

        fig = plt.figure(3, figsize=(9, 9), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_xlim(min(x)-0.5,max(x)+0.5)
        ax.set_ylim(min(y) - 0.5, max(y) + 0.5)

        ax.plot(x, y, 'x', self.splineShape[0], self.splineShape[1], 'b')
        out = np.array(self.splineShape_s)
        x = out[:, 0]
        y = out[:, 1]
        ax.plot(x, y, color = 'r')
        ax.annotate("Start", xy=(2, 2), xytext=(x[0], y[0])) #,arrowprops=dict(facecolor='black', shrink=0.05))

        plt.title('Points defining the natural cell shape')
        plt.show()

    def calculate_meridionalCurvature(self, show=False):
        ''' Calculates the meridional curvature in the interval (a,b) '''

        self.kappa_m = []

        for i in range(0, len(self.splineShape_s)):
            sign = 1.0
            if (self.Dspline_Du_s[i][0] < 0.0):
                sign = -1.0
            nominator = - sign * (
                self.D2spline_D2u_s[i][1] * self.Dspline_Du_s[i][0] - self.D2spline_D2u_s[i][0] * self.Dspline_Du_s[i][1])
            denominator = np.sqrt(self.Dspline_Du_s[i][0] ** 2 + self.Dspline_Du_s[i][1] ** 2) ** 3
            temp = nominator / denominator
            self.kappa_m.append([self.splineShape_s[i][0], temp, self.splineShape_s[i][2]])

        # print self.kappa_m
        # print len(self.splineShape_upHalf)

        # for i in range(1, len(self.splineShape[0]) - 1):
        #    if ((a < self.splineShape[0][i]) & (self.splineShape[0][i] < b) & (self.splineShape[1][i] > 0.0)):
        #        sign = 1.0
        #        if (self.Dspline_Du[0][i] < 0.0):
        #            sign = -1.0
        #        nominator = - sign * (
        #        self.D2spline_D2u[1][i] * self.Dspline_Du[0][i] - self.D2spline_D2u[0][i] * self.Dspline_Du[1][i])
        #        denominator = np.sqrt(self.Dspline_Du[0][i] ** 2 + self.Dspline_Du[1][i] ** 2) ** 3
        #        temp = nominator / denominator
        #        self.kappa_m.append([self.splineShape[0][i], temp])
        # print temp

        if (show):
            out = np.array(self.kappa_m)
            x = out[:, 0]
            y = out[:, 1]

            plt.figure()
            plt.plot(x, y, 'x', self.splineShape[0], self.splineShape[1], 'b')
            plt.title('Meridional curvature along upper half of the shape')
            plt.show()

    def calculate_circumferentialCurvature(self, show=False):
        ''' Calculates the meridional curvature in the interval (a,b) '''

        self.kappa_theta = []

        for i in range(0, len(self.splineShape_s)):
            nominator = self.Dspline_Du_s[i][0]
            denominator = np.absolute(self.splineShape_s[i][1]) * np.sqrt(
                self.Dspline_Du_s[i][0] ** 2 + self.Dspline_Du_s[i][1] ** 2)
            temp = np.abs(nominator / denominator)
            self.kappa_theta.append([self.splineShape_s[i][0], temp, self.splineShape_s[i][2]])

            # for i in range(1, len(self.splineShape[0]) - 1):
            #    if ((a < self.splineShape[0][i]) & (self.splineShape[0][i] < b) & (self.splineShape[1][i] > 0.0)):
            #        nominator = self.Dspline_Du[0][i]
            #        denominator = np.absolute(self.splineShape[1][i]) * np.sqrt(
            #            self.Dspline_Du[0][i] ** 2 + self.Dspline_Du[1][i] ** 2)
            #        temp = nominator / denominator
            #        self.kappa_theta.append([self.splineShape[0][i], temp])
            # print temp

        out = np.array(self.kappa_theta)

        x = out[:, 0]
        y = out[:, 1]

        if (show):
            plt.figure()
            plt.plot(x, y, 'x', self.splineShape[0], self.splineShape[1], 'b')
            plt.title('Circumferential curvature along upper half of the shape')
            plt.show()
