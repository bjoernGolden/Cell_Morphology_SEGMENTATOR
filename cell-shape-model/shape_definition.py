import numpy as np
import matplotlib.pyplot as plt
import csv


class Shape_definition:
    pointsNaturalShape = []
    pointsRelaxedShape = []

    def __init__(self, params):

        self.pointsNaturalShape = []
        self.pointsRelaxedShape = []
        #self.construct_naturalShape(params)
        self.read_shape('natural_shape_points.csv')
        self.construct_relaxedShape(params)
        self.mirror_along_xAxis()
        self.shift_start_point()
        self.reverse_order()
        self.close_curve()


    def construct_naturalShape(self, params):
        ''' Construct geometry '''

        # Points for spline fitting

        ## Construct Base

        circle_sections = 10

        for i in range(0, circle_sections):
            theta = (np.pi / (1.7 * circle_sections)) * i
            # self.pointsNaturalShape.append([params.r_base * (1.0 - np.cos(np.pi / 1.7 - theta)), params.r_base * np.sin(np.pi / 1.7 - theta)])
            self.pointsNaturalShape.append([params.r_base * (1.0 - np.cos(theta)), params.r_base * np.sin(theta)])

        ## Transition region


        self.pointsNaturalShape.append([params.r_base + 1.4, params.r_base - 0.3])

        ## Construct Shaft

        sections = 3

        shaft_end = params.L - params.r_tip

        shaft_start = params.r_base + params.transistion_length

        for i in range(0, sections):
            lamda = (1.0 / sections) * i
            self.pointsNaturalShape.append([(1.0 - lamda) * shaft_start + lamda * shaft_end,
                                            (1.0 - lamda) * params.r_shaft + lamda * params.r_tip])

        self.pointsNaturalShape[-2][1] -= 0.05

        ## Construct tip with radius r_tip

        circle_sections = 5

        # for i in range(2, circle_sections):
        ##theta = (np.pi / (2.0 * circle_sections)) * i
        # theta = (np.pi / (2.0 * circle_sections)) * i
        ##self.pointsNaturalShape.append([params.L - params.r_tip * (1.0 - np.sin(theta)), params.r_tip * np.cos(theta)])
        # self.pointsNaturalShape.append([params.L - params.r_base * (1.0 - np.sin(theta)), params.r_base * np.cos(theta)])

        theta = (np.pi / (2.0 * circle_sections)) * 2
        self.pointsNaturalShape.append(
            #[params.L - params.r_tip * (1.0 - np.sin(theta)), params.r_tip * np.cos(theta) + 0.05])
            [params.L - params.r_tip * (1.0 - np.sin(theta)), params.r_tip * np.cos(theta)])

        theta = (np.pi / (2.0 * circle_sections)) * 3
        self.pointsNaturalShape.append(
            [params.L - params.r_tip * (1.0 - np.sin(theta)) - 0.15, params.r_tip * np.cos(theta)])

        theta = (np.pi / (2.0 * circle_sections)) * 4
        self.pointsNaturalShape.append(
            [params.L - params.r_tip * (1.0 - np.sin(theta)) - 0.21, params.r_tip * np.cos(theta)])

        # theta = (np.pi / (2.0 * circle_sections)) * 5
        ##self.pointsNaturalShape.append([params.L - params.r_tip * (1.0 - np.sin(theta)), params.r_tip * np.cos(theta)])
        # self.pointsNaturalShape.append([params.L - params.r_tip * (1.0 - np.sin(theta))-0.23, params.r_tip * np.cos(theta)])

    def construct_relaxedShape(self, params):
        phi = np.arange(0.0, 0.8 * np.pi, 0.1)

        '''upper base part '''
        for i in range(0, len(phi)):
            self.pointsRelaxedShape.append(
                [params.r_base - params.R_base * np.cos(phi[i]), params.R_base * np.sin(phi[i])])

        ''' upper shaft part '''

        temp_x = np.arange(params.r_base + params.R_base + 0.8, params.L - 1.5, 0.01)
        for i in range(0, len(temp_x) - 1):
            self.pointsRelaxedShape.append([temp_x[i], params.R_shaft])

        self.pointsRelaxedShape.append([temp_x[-1] + 1.0, params.R_shaft / 2.0])

        ''' Take relaxed shape as a shrinked version of the natural shape'''
        # temp = []
        # shrinkfactor = params.R_base / params.r_base
        # for i in range(0, len(self.pointsNaturalShape)):
        # self.pointsRelaxedShape.append([self.pointsNaturalShape[i][0] * shrinkfactor + params.r_base - params.R_base,
        # self.pointsNaturalShape[i][1] * shrinkfactor])

    def mirror_along_xAxis(self):

        temp = len(self.pointsNaturalShape) - 1
        for i in range(0, len(self.pointsNaturalShape)):
            x = self.pointsNaturalShape[temp - i][0]
            y = -self.pointsNaturalShape[temp - i][1]
            if (y != 0):
                self.pointsNaturalShape.append([x, y])

        temp = len(self.pointsRelaxedShape) - 1
        for i in range(0, len(self.pointsRelaxedShape)):
            x = self.pointsRelaxedShape[temp - i][0]
            y = -self.pointsRelaxedShape[temp - i][1]
            if (y != 0):
                self.pointsRelaxedShape.append([x, y])

    def shift_start_point(self):

        num_points = len(self.pointsNaturalShape)
        min_y = self.pointsNaturalShape[0][1]
        min_y_index = 0

        for i in range(1, num_points):
            if (min_y > self.pointsNaturalShape[i][1]):
                min_y = self.pointsNaturalShape[i][1]
                min_y_index = i

        temp_x = []
        temp_y = []

        for i in range(0, num_points):
            temp_x.append(self.pointsNaturalShape[(i + min_y_index) % num_points][0])
            temp_y.append(self.pointsNaturalShape[(i + min_y_index) % num_points][1])

        for i in range(0, num_points):
            self.pointsNaturalShape[i][0] = temp_x[i]
            self.pointsNaturalShape[i][1] = temp_y[i]

        # self.pointsNaturalShape.append([temp_x[0],temp_y[0]])


        num_points = len(self.pointsRelaxedShape)
        min_y = self.pointsRelaxedShape[0][1]
        min_y_index = 0

        for i in range(1, num_points):
            if (min_y > self.pointsRelaxedShape[i][1]):
                min_y = self.pointsRelaxedShape[i][1]
                min_y_index = i

        temp_x = []
        temp_y = []

        for i in range(0, num_points):
            temp_x.append(self.pointsRelaxedShape[(i + min_y_index) % num_points][0])
            temp_y.append(self.pointsRelaxedShape[(i + min_y_index) % num_points][1])

        for i in range(0, num_points):
            self.pointsRelaxedShape[i][0] = temp_x[i]
            self.pointsRelaxedShape[i][1] = temp_y[i]

            # self.pointsNaturalShape.append([temp_x[0], temp_y[0]])

    def reverse_order(self):

        num_points = len(self.pointsNaturalShape)

        temp_x = []
        temp_y = []

        for i in range(0, num_points):
            temp_x.append(self.pointsNaturalShape[i][0])
            temp_y.append(self.pointsNaturalShape[i][1])

        for i in range(0, num_points):
            self.pointsNaturalShape[i][0] = temp_x[num_points - i - 1]
            self.pointsNaturalShape[i][1] = temp_y[num_points - i - 1]

        num_points = len(self.pointsRelaxedShape)

        temp_x = []
        temp_y = []

        for i in range(0, num_points):
            temp_x.append(self.pointsRelaxedShape[i][0])
            temp_y.append(self.pointsRelaxedShape[i][1])

        for i in range(0, num_points):
            self.pointsRelaxedShape[i][0] = temp_x[num_points - i - 1]
            self.pointsRelaxedShape[i][1] = temp_y[num_points - i - 1]

    def close_curve(self):

        temp_x = self.pointsNaturalShape[0][0]
        temp_y = self.pointsNaturalShape[0][1]
        if ((temp_x != self.pointsNaturalShape[-1][0]) or (temp_y != self.pointsNaturalShape[-1][1])):
            self.pointsNaturalShape.append([temp_x, temp_y])

        temp_x = self.pointsRelaxedShape[0][0]
        temp_y = self.pointsRelaxedShape[0][1]
        if ((temp_x != self.pointsRelaxedShape[-1][0]) or (temp_y != self.pointsRelaxedShape[-1][1])):
            self.pointsRelaxedShape.append([temp_x, temp_y])

    def show_pointsNaturalShape(self):
        ''' Plot points '''

        temp_points = np.array(self.pointsNaturalShape)
        x = temp_points[:, 0]
        y = temp_points[:, 1]

        fig = plt.figure(3, figsize=(9, 9), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # plt.figure()
        for i in range(0, len(x)):
            ax.plot(x[i], y[i], 'x')
            ax.annotate(str(i), xy=(1, 1), xytext=(x[i], y[i]))
        ax.plot(x, y, 'b')
        plt.title('Points defining the natural cell shape')
        plt.show()

    def relaxed_shape(self, x, params):
        ''' Definition of the relaxed shape '''

        # y is set to R_shaft in the growth region of the relaxed shape
        y = params.R_shaft

        # y lies on a half circle of radius r_base in the base part of the relaxed shape
        if ((params.r_base - params.R_base < x) & (x <= params.R_base + params.r_base)):
            temp = np.sqrt(params.R_base ** 2 - (x - params.r_base) ** 2)
            y = temp
            if (y < params.R_shaft):
                y = params.R_shaft
        # y is set to 0.0 on the left of the base part of the relaxed shape
        if (x <= params.r_base - params.R_base):
            y = 0.0

        return

    def show_relaxedShape(self, params):
        ''' Plot relaxed shape '''

        # x = np.arange(0.0, params.L, 0.1)
        # y = []

        # for i in range(0, len(x)):
        #   y.append(self.relaxed_shape(x[i], params))

        temp_points = np.array(self.pointsRelaxedShape)
        x = temp_points[:, 0]
        y = temp_points[:, 1]

        fig = plt.figure(3, figsize=(9, 9), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # plt.figure()
        for i in range(0, len(x)):
            ax.plot(x[i], y[i], 'x')
            ax.annotate(str(i), xy=(1, 1), xytext=(x[i], y[i]))
        ax.plot(x, y, 'b')
        plt.title('Points defining the natural cell shape')
        plt.show()

    def show_bothShapes(self, params):

        temp_points = np.array(self.pointsNaturalShape)
        x1 = temp_points[:, 0]
        y1 = temp_points[:, 1]

        temp_points = np.array(self.pointsRelaxedShape)
        x2 = temp_points[:, 0]
        y2 = temp_points[:, 1]

        fig = plt.figure(3, figsize=(9, 9), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # plt.figure()
        for i in range(0, len(x1)):
            ax.plot(x1[i], y1[i], 'x')
            ax.annotate(str(i), xy=(1, 1), xytext=(x1[i], y1[i]))
        ax.plot(x1, y1, 'b')

        # plt.figure()
        for i in range(0, len(x2)):
            ax.plot(x2[i], y2[i], 'x')
            ax.annotate(str(i), xy=(1, 1), xytext=(x2[i], y2[i]))
        ax.plot(x2, y2, 'b')
        plt.title('Natural and Relaxed Cell Shape')
        plt.show()

    def arclength_S_to_yS(self, arc_S, params):
        ''' calculate y postion corresponding to given arclength arc_S,
        the arclength is meashured from the beginning of the base part of the relaxed cel shape'''

        theta = np.arcsin(params.R_shaft / params.R_base)
        transition_S = params.R_base * (np.pi - theta)
        yS_ = params.R_shaft
        if (arc_S < transition_S):
            phi = arc_S / params.R_base
            yS_ = params.R_base * np.sin(phi)
        return yS_

    def arclength_S_to_xS(self, arc_S, params):
        ''' calculate x postion corresponding to given arclength arc_S,
        the arclength is meashured from the beginning of the base part of the relaxed cel shape'''

        theta = np.arcsin(params.R_shaft / params.R_base)
        transition_S = params.R_base * (np.pi - theta)
        xS_ = 0.0
        if (arc_S < transition_S):
            phi = arc_S / params.R_base
            xS_ = params.r_base - params.R_base * np.cos(phi)
        else:
            xS_ = params.r_base + params.R_base * np.cos(theta) + arc_S - transition_S

        return xS_

    def write_shape(self, filename):

        out = csv.writer(open('natural_shape_points.csv', 'w'), delimiter=',', quoting=csv.QUOTE_ALL)

        return 0

    def read_shape(self,filename):

        self.pointsNaturalShape = []
        input_file = open(filename, 'r')
        data = csv.reader(input_file)
        counter = 0
        for line in data:
            if (counter > 0):
                self.pointsNaturalShape.append([float(line[0]),float(line[1])])
            counter +=1

        del data
        input_file.close()

        return 0
