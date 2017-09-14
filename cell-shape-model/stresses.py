import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class Stresses:
    ''' Calculate and store stresses '''

    naturalShape = []

    def __init__(self, spline, params):
        self.sigma_m = []  # meridional stress
        self.sigma_theta = []  # circumferential stress
        self.sigma_max = []  # maximum stress
        self.sigma_vM = []  # von Mises stress

        self.naturalShape = spline.splineShape
        self.params = params

        self.calculate_meridionalStress(params, spline.kappa_theta, True)
        self.calculate_circumferentialStress(params, spline.kappa_theta, spline.kappa_m, True)


    def calculate_meridionalStress(self, params, kappa_theta, show=False):
        ''' Claculate the meridional stress '''

        self.sigma_m = []
        self.params = params

        for i in range(0, len(kappa_theta)):
            self.sigma_m.append([kappa_theta[i][0], params.P / (2.0 * params.h * kappa_theta[i][1]), kappa_theta[i][2]])

        if (show):
            self.show_stressDistribution(self.sigma_m, 'Meridional stress along the upper half of the shape')

    def calculate_circumferentialStress(self, params, kappa_theta, kappa_m, show=False):
        ''' Claculate the circumferential stress '''

        self.sigma_theta = []

        for i in range(0, len(kappa_theta)):
            self.sigma_theta.append([kappa_theta[i][0], (2.0 - kappa_m[i][1] / kappa_theta[i][1]) * params.P / (
                2.0 * params.h * kappa_theta[i][1]), kappa_theta[i][2]])

        if (show):
            self.show_stressDistribution(self.sigma_theta, 'Circumferential stress along the upper half of the shape')

    def calculate_maximalStress(self, show=False):
        ''' clalculate the maximal stress '''

        self.sigma_max = []
        for i in range(0, len(self.sigma_m)):
            self.sigma_max.append([self.sigma_m[i][0], max(self.sigma_m[i][1], self.sigma_theta[i][1])])

        if (show):
            self.show_stressDistribution(self.sigma_max,
                                         'Maximum stress distribution along the upper half of the shape')

    def calculate_vonMisesStress(self, show=False):
        ''' clalculate the maximal stress '''

        self.sigma_vM = []
        for i in range(0, len(self.sigma_m)):
            self.sigma_vM.append([self.sigma_m[i][0], np.sqrt(
                self.sigma_m[i][1] ** 2 + self.sigma_theta[i][1] ** 2 - self.sigma_m[i][1] * self.sigma_theta[i][1])])

        if (show):
            self.show_stressDistribution(self.sigma_vM,
                                         'von Mises stress distribution along the upper half of the shape')

    def show_stressDistribution(self, sigma, title):
        ''' plots a given stress distribution '''

        #out = np.array(sigma)
        #x = out[:, 0]
        #y = out[:, 1]

        #plt.figure()
        #plt.plot(x, y, 'x', self.naturalShape[0], self.naturalShape[1], 'b')
        #plt.title(title)
        #plt.show()

        out = np.array(sigma)
        x = out[:, 0]
        y = out[:, 1]
        s = out[:, 2]


        plt.figure()
        plt.plot(x, y, 'x', self.naturalShape[0], self.naturalShape[1], 'b')
        plt.title(title)
        plt.show()

        # plt.figure()
        # plt.plot(s, y, 'x', self.naturalShape[0], self.naturalShape[1], 'b')
        # plt.title(title + str("(arclength)"))
        # plt.show()

        stress_value = self.params.P * self.params.r_base / (2.0 * self.params.h)

        fig = plt.figure(3, figsize=(9, 5), dpi=300)

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_ylim(-0.3, 4.7)
        ax.set_xlim(0.0, 10.1)

        ax.plot(s, y, '-')
        ax.tick_params(length=4, width=1.0, size=3, labelsize=8)
        xtick_locs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        xtick_lbls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        plt.xticks(xtick_locs, xtick_lbls)

        plt.title("Stress Distribution", fontsize=12)
        plt.savefig(title + '.eps', format="eps")
        plt.savefig(title + '.pdf', format="pdf")
        plt.show()

    def show_stressDistribution3D(self, sigma, title):

        # plt.figure()
        # plt.plot(s, y, 'x', self.spline[0], self.spline[1], 'b')
        # plt.title('Circumferential curvature along arclength')
        # plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        naturalShape_upperHalf = []

        for i in range(1, len(self.naturalShape[0]) - 1):
            if ((self.sigma_m[0][0] <= self.naturalShape[0][i]) & (
                        self.naturalShape[0][i] <= self.sigma_m[-1][0]) & (self.naturalShape[1][i] > 0.0)):
                naturalShape_upperHalf.append([self.naturalShape[0][i], self.naturalShape[1][i]])

        theta_grid = np.linspace(0, 2 * np.pi, 100)
        out = np.array(naturalShape_upperHalf)
        shape_x = out[:, 0]
        shape_y = out[:, 1]

        r_points, theta_points = np.meshgrid(shape_y, theta_grid)

        out = np.array(sigma)
        sigma_x = out[:, 0]
        sigma_s = out[:, 1]
        sigma_y = out[:, 2]
        data, theta_points = np.meshgrid(sigma_y, theta_grid)

        x_points, y_points = r_points * np.cos(theta_points), r_points * np.sin(theta_points)

        ax.plot_surface(x_points, y_points, shape_x, rstride=1, cstride=1, cmap=cm.YlGnBu_r)

        # plt.show()

        sigma_range = max(sigma_y) - min(sigma_y)
        # sigma_max = 3.0


        # N = data/data.max()
        # N = data/sigma_range

        max_base = 0.0
        max_trans = 0.0
        max_shaft = 0.0
        max_tip = 0.0
        min_base = 100.0
        min_trans = 100.0
        min_shaft = 100.0
        min_tip = 100.0

        for i in range(0, len(sigma)):

            if (sigma[i][0] < self.params.r_base):
                if (max_base < sigma[i][2]):
                    max_base = sigma[i][2]
                if (min_base > sigma[i][2]):
                    min_base = sigma[i][2]

            if ((self.params.r_base < sigma[i][0]) & (
                        sigma[i][0] < self.params.L - self.params.r_shaft - self.params.r_tip)):
                if (max_trans < sigma[i][2]):
                    max_trans = sigma[i][2]
                if (min_trans > sigma[i][2]):
                    min_trans = sigma[i][2]

            if ((self.params.L - self.params.r_shaft - self.params.r_tip < sigma[i][0]) & (
                        sigma[i][0] < self.params.L - self.params.r_tip)):
                if (max_shaft < sigma[i][2]):
                    max_shaft = sigma[i][2]
                if (min_shaft > sigma[i][2]):
                    min_shaft = sigma[i][2]

            if (self.params.L - self.params.r_tip < sigma[i][0]):
                if (max_tip < sigma[i][2]):
                    max_tip = sigma[i][2]
                if (min_shaft > sigma[i][2]):
                    min_tip = sigma[i][2]

        print title
        print "Region\tMax\tMin"
        print "Global\t" + str(max(sigma_y)) + "\t" + str(min(sigma_y))
        print "Base\t" + str(max_base) + "\t" + str(min_base)
        print "Neck\t" + str(max_trans) + "\t" + str(min_trans)
        print "Shaft\t" + str(max_shaft) + "\t" + str(min_shaft)
        print "Tip\t" + str(max_tip) + "\t" + str(min_tip)

        N = data / 2.0

        # print N
        # print cm.jet(N)

        ax.plot_surface(x_points, y_points, shape_x, rstride=1, cstride=1, facecolors=cm.coolwarm(N), linewidth=1,
                        antialiased=False, shade=False)
        # ax.set_zlim3d(0, cell_radius + shmoo_length)
        # ax.set_xlabel(r'$x$')
        # ax.set_ylabel(r'$y$')
        # ax.set_zlabel(r'$z$')

        ax.grid(False)
        for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
            for t in a.get_ticklines() + a.get_ticklabels():
                t.set_visible(False)
            a.line.set_visible(False)
            a.pane.set_visible(False)

        m = cm.ScalarMappable(cmap=cm.coolwarm)
        m.set_array(N)
        m.set_clim(0.0, 2.0)
        plt.colorbar(m)
        plt.savefig("Stress.tif")
        plt.savefig("Stress.eps", format="eps")
        plt.show()
