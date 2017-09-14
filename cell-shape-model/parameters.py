import numpy as np



class Params:
    ''' User parameters are defined here  '''

    def __init__(self):
        ''' Parameters Physics '''
        self.P = 0.2
        self.h = 0.12
        self.nu = 0.5

        ''' Parameters Shape '''
        self.R_base = 1.9
        self.r_base = 2.5
        self.R_shaft = 0.5
        self.r_shaft = 1.7
        self.r_tip = 0.7
        self.tipgrowth_radius = 0.2
        self.L = 8.0
        self.transistion_length = 2.2

        ''' Definition of Regions '''

        #self.s_tip = 1.25 # for arclength plot
        self.s_tip = 0.65
        self.s_neck = 3.0
        self.s_base = 5.0

        ''' Output/Numerical Parameters '''

        self.interval = [0.1, 8.0]
        self.elasticityInterval = [0.5, 7.7]
        self.breakThreshold_epsilonTheta = 0.1
        self.breakThreshold_E = 0.5
        self.epsilon = 0.2
        self.smoothness_natural = 0.01
        self.smoothness_relaxed = 0.01
        #fine
        self.step_size_natural = 0.00004
        self.step_size_relaxed = 0.00001
        #coarse
        #self.step_size_natural = 0.0004
        #self.step_size_relaxed = 0.0001
        #very coarse
        #self.step_size_natural = 0.04
        #self.step_size_relaxed = 0.01
        self.start_phi = np.pi/4.0

    def set_Rshaft(self,R):
	self.R_shaft = R

