import numpy as np
import matplotlib.pyplot as plt


class Strain:
    ''' Calculate and store strains '''

    naturalShape = []
    relaxedShape = []
    # arclength_s = []
    # arclength_S = []
    #    kappa_theta = []
    #    kappa_m = []
    strain_m = []  # meridional stress
    strain_theta = []  # circumferential stress
    dS = []  # maximum stress
    ds = []
    E_theta = []
    E_m = []

    def __init__(self, spline_natural, spline_relaxed, params):
        '''  '''
        self.naturalShape = []
        self.relaxedShape = []
        # self.arclength_s = []
        # self.arclength_S = []
        self.strain_m = []  # meridional stress
        self.strain_theta = []  # circumferential stress
        self.dS = []  # maximum stress
        self.ds = []
        self.E_theta = []
        self.E_m = []

        self.naturalShape = spline_natural.splineShape_s
        self.relaxedShape = spline_relaxed.splineShape_s

        self.calculate_strains(spline_natural, spline_relaxed, params)
        #self.show_strains()
        #self.show_elasticity()

    def S_to_index(self, S):

        index = -1

        for i in range(0, len(self.relaxedShape) - 1):
            if ((self.relaxedShape[i][2] <= S) & (S < self.relaxedShape[i + 1][2])):
                index = i

        return index

    def calculate_strains(self, spline_natural, spline_relaxed, params):
        ''' calculate strains'''

        self.naturalShape = spline_natural.splineShape_s
        self.relaxedShape = spline_relaxed.splineShape_s

        index_S = 0
        arclength_s = 0.0
        arclength_S = 0.0

        for i in range(0, len(self.naturalShape) - 1):

            arclength_s = self.naturalShape[i][2]

            index_S = self.S_to_index(arclength_S)

            #print "index_S"
            #print index_S

            if (index_S < 0):
                print "Index_S out of range! End of arclength for relaxed shape reached!"
                break

            ds_value = self.naturalShape[i + 1][2] - self.naturalShape[i][2]

            xS_ = self.relaxedShape[index_S][0]
            yS_ = self.relaxedShape[index_S][1]

            epsilon_theta = (self.naturalShape[i][1] / yS_ - 1.0)

            if (epsilon_theta < params.breakThreshold_epsilonTheta ):
                print "Break Threshold reached for circumferential strain"
                break

            #print "epsilon_theta"
            #print epsilon_theta
            #print "ys"
            #print self.naturalShape[i][1]
            #print "yS"
            #print yS_

            nominator = (1.0 - 2.0 * params.nu) * spline_natural.kappa_theta[i][1] + params.nu * spline_natural.kappa_m[i][1]
            denominator = (2.0 - params.nu) * spline_natural.kappa_theta[i][1] - spline_natural.kappa_m[i][1]
            epsilon_m = epsilon_theta * nominator / denominator

            dS_value0 = ds_value / (epsilon_m + 1.0)

            #print "dS_value"
            #print dS_value0

            self.strain_m.append([arclength_s, epsilon_m, spline_natural.kappa_theta[i][0]])
            self.strain_theta.append([arclength_s, epsilon_theta, spline_natural.kappa_theta[i][0]])

            arclength_S += dS_value0
            # self.arclength_S.append(dS_sum)

            sigma_m = params.P / (2.0 * params.h * spline_natural.kappa_theta[i][1])
            sigma_theta = sigma_m * (2.0 - spline_natural.kappa_m[i][1] / spline_natural.kappa_theta[i][1])

            E = (sigma_theta - params.nu * sigma_m) / epsilon_theta
            E = E * (1.0 - params.nu * params.nu)
            self.E_theta.append([arclength_s, E, spline_natural.kappa_theta[i][0]])
            # print "E_theta: " + str(E)



            E = (sigma_m - params.nu * sigma_theta) / epsilon_m
            E = E*(1.0 - params.nu*params.nu)
            self.E_m.append([arclength_s, E, spline_natural.kappa_theta[i][0]])  # Arclength, E value, x coordinate

            #if (E < params.breakThreshold_E):
                #print "Break Threshold for Young's Modulus reached!"
                #break

        print "Meridional Strain"
        print self.strain_m
        print "Circumfernatial Strain"
        print self.strain_theta

    def show_strains(self):

        out = np.array(self.strain_m)
        s = out[:, 0]
        y_m = out[:, 1]

        # print "Arclength: "
        # print s
        # print "Meridional Strain"
        # print y_m

        out2 = np.array(self.strain_theta)
        y_theta = out2[:, 1]
        # print "Arclength: "
        # print s
        # print "Circumferential Strain"
        # print y_theta



        fig = plt.figure(3, figsize=(9, 3), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.set_ylim(-0.3, 2.7)
        # ax.set_xlim(0.0, 9.1)

        # plt.figure()
        ##plt.plot(s, y_m, 'x', s, y_theta, 'o', spline_natural[0], spline_natural[1], 'b')
        # plt.title('Meridional and Circumferential Strain')
        # plt.ylim(-3.0,3.0)
        # plt.show()

        #ax.plot(s, y_m, '-', s, y_theta, '-')
        ax.plot(s, y_m, '-')
        ax.plot(s, y_theta, '-')

        ax.tick_params(length=4, width=1.0, size=3, labelsize=8)
        xtick_locs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        xtick_lbls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        plt.xticks(xtick_locs, xtick_lbls)
        # xtick_lbls = []
        # plt.xticks(xtick_locs, xtick_lbls)
        # ytick_locs = [100,200,300,400,500,600]
        # ytick_lbls = []
        # plt.yticks(ytick_locs, ytick_lbls)

        plt.title('Meridional and Circumferential Strain', fontsize=12)
        plt.savefig('meridional_n_circumferential_strain.eps', format="eps")
        plt.savefig('meridional_n_circumferential_strain.pdf', format="pdf")
        plt.show()

    def show_elasticity(self):

        out = np.array(self.E_m)
        s = out[:, 0]
        y_m = out[:, 1]

        out2 = np.array(self.E_theta)
        y_theta = out2[:, 1]

        # plt.figure()
        # plt.plot(s, y_theta, 'o', spline_natural[0], spline_natural[1], 'b')
        # plt.plot(s, -y_m, 'x')
        # plt.title('Youngs modulus along arclength')
        # plt.ylim(-5.0,5.0)


        fig = plt.figure(3, figsize=(9, 6), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_ylim(-0.3, 5.7)
        ax.set_xlim(0.0, 9.1)

        ax.plot(s, y_m, '-', s, y_theta, '-')
        ax.tick_params(length=4, width=1.0, size=3, labelsize=8)
        xtick_locs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        xtick_lbls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        plt.xticks(xtick_locs, xtick_lbls)

        plt.title("Young's Modulus", fontsize=12)
        plt.savefig('E_arclength.eps', format="eps")
        plt.savefig('E_arclength.pdf', format="pdf")
        plt.show()


        # ds_sum = self.arclength_s[-1]
        # dS_sum = self.arclength_S[-1]

        # for i in range(0, len(self.naturalShape) - 1):
        # ds_value = np.sqrt((self.naturalShape[i][0] - self.naturalShape[i + 1][0]) ** 2 + (
        # self.naturalShape[i][1] - self.naturalShape[i + 1][1]) ** 2)
        # ds_sum += ds_value
        # self.ds.append(ds_value)
        # self.arclength_s.append(ds_sum)

        ## xS_ = shape_def.arclength_S_to_xS(self.arclength_S[-1],params)
        ## yS_ = shape_def.arclength_S_to_yS(self.arclength_S[-1],params)
        # xS_ = self.relaxedShape[-1][0]
        # yS_ = self.relaxedShape[-1][1]

        # epsilon_theta = (self.naturalShape[i][1] / yS_ - 1.0)

        # nominator = (1.0 - 2.0 * params.nu) * self.kappa_theta[i][1] + params.nu * self.kappa_m[i][1]
        # denominator = (2.0 - params.nu) * self.kappa_theta[i][1] - self.kappa_m[i][1]
        # epsilon_m = epsilon_theta * nominator / denominator + 1.0

        # dS_value0 = ds_value / epsilon_m

        # self.strain_m.append(epsilon_m)
        # self.strain_theta.append(epsilon_theta)

        # dS_sum += dS_value0
        # self.arclength_S.append(dS_sum)

        # xS = shape_def.arclength_S_to_xS(self.arclength_S[-1], params)
        # yS = shape_def.arclength_S_to_yS(self.arclength_S[-1], params)

        # self.relaxedShape.append([xS, yS])

    def show_pointIdentification(self, params, jumps):
        ''' show the identification of points in the new and the old shape '''

        fig = plt.figure(3, figsize=(3, 3), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        ax.set_ylim(-4.1, 4.1)
        ax.set_xlim(-0.2, 8.0)

        naturalShape_points = []
        relaxedShape_points = []

        for i in range(0, len(self.naturalShape) / jumps):
            naturalShape_points.append(self.naturalShape[i * jumps])
            relaxedShape_points.append(self.relaxedShape[i * jumps])

        ## Reconstruction of the base end. This is necessary, since the curvature can not be calculated at the very end of the shape

        phi = np.arctan2(self.naturalShape[0][1], params.r_base - self.naturalShape[0][0])

        step = np.sqrt((naturalShape_points[0][0] - naturalShape_points[1][0]) ** 2 + (
            naturalShape_points[0][1] - naturalShape_points[1][1]) ** 2)
        step = step / params.r_base
        # step = step/1.02

        theta = np.arange(-phi, phi, step);

        ax.plot([params.r_base * (1.0 - np.cos(theta)), params.r_base - params.R_base * np.cos(theta)],
                [-params.r_base * np.sin(theta), -params.R_base * np.sin(theta)], '-', color='black')

        ax.plot(params.r_base * (1.0 - np.cos(theta)), -params.r_base * np.sin(theta), 'o', color="red", markersize=2.5)
        ax.plot(params.r_base - params.R_base * np.cos(theta), -params.R_base * np.sin(theta), 'o', color="blue",
                markersize=2.5)

        ax.plot(params.r_base * (1.0 - np.cos(theta)), -params.r_base * np.sin(theta), '-', color="red")
        ax.plot(params.r_base - params.R_base * np.cos(theta), -params.R_base * np.sin(theta), '-', color="blue")

        out = np.array(naturalShape_points)
        nat_shape_x = out[:, 0]
        nat_shape_y = out[:, 1]

        out = np.array(relaxedShape_points)
        rel_shape_x = out[:, 0]
        rel_shape_y = out[:, 1]

        # plt.figure()
        ax.plot(nat_shape_x, nat_shape_y, 'o', color="red", markersize=2.5)
        ax.plot(rel_shape_x, rel_shape_y, 'o', color="blue", markersize=2.5)

        # ax.plot([self.naturalShape[-1][0]],[self.naturalShape[-1][1]],'o', [self.relaxedShape[-1][0]],[self.relaxedShape[-1][1]], 'o')

        for i in range(0, len(rel_shape_x)):
            ax.plot([nat_shape_x[i], rel_shape_x[i]], [nat_shape_y[i], rel_shape_y[i]], '-', color="black")

        # ax.plot([self.naturalShape[-1][0],self.relaxedShape[-1][0]], [self.naturalShape[-1][1],self.relaxedShape[-1][1]], '-', color="black")
        for i in range(0, len(nat_shape_y)):
            nat_shape_y[i] = - nat_shape_y[i]
            rel_shape_y[i] = - rel_shape_y[i]

        ax.plot(nat_shape_x, nat_shape_y, 'o', color="red", markersize=2.5)
        ax.plot(rel_shape_x, rel_shape_y, 'o', color="blue", markersize=2.5)

        # plot last point
        # ax.plot([self.naturalShape[-1][0]],[self.naturalShape[-1][1]],'o', [self.relaxedShape[-1][0]],[self.relaxedShape[-1][1]], 'o')

        for i in range(0, len(rel_shape_x)):
            ax.plot([nat_shape_x[i], rel_shape_x[i]], [nat_shape_y[i], rel_shape_y[i]], '-', color="black")

        out = np.array(self.naturalShape)
        nat_shape_x = out[:, 0]
        nat_shape_y = out[:, 1]

        out = np.array(self.relaxedShape)
        rel_shape_x = out[:, 0]
        rel_shape_y = out[:, 1]

        ax.plot(nat_shape_x, nat_shape_y, '-', color="red")
        ax.plot(rel_shape_x, rel_shape_y, '-', color="blue")

        for i in range(0, len(nat_shape_y)):
            nat_shape_y[i] = - nat_shape_y[i]
            rel_shape_y[i] = - rel_shape_y[i]

        ax.plot(nat_shape_x, nat_shape_y, '-', color="red")
        ax.plot(rel_shape_x, rel_shape_y, '-', color="blue")

        ax.tick_params(length=4, width=1.0, size=3, labelsize=8)
        # xtick_locs = [1.5,3.0,4.5,6.0,9.0,12.0,15.0]
        # xtick_lbls = []
        # plt.xticks(xtick_locs, xtick_lbls)
        # ytick_locs = [100,200,300,400,500,600]
        # ytick_lbls = []
        # plt.yticks(ytick_locs, ytick_lbls)

        plt.title("Point identification", fontsize=12)
        plt.savefig("point_identification.eps", format="eps")
        plt.show()
