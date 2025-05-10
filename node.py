# node class

from numpy import array, zeros
#from numba import jit, jitclass, int32, float64
#spec = [
#        ('dof', int32),
#        ('referencePosition', float64[:]),
#        ('position', float64[:]),
#        ('displacement', float64[:]),
#        ('forces', float64[:]),
#        ('alphaNM', float64),
#        ('betaNM', float64),
#        ('timeStep', int32),
#        ('sigma', float64[:]),
#        ('weightFactor', float64),
#        ('vonMises', float64),
#        ('uPrev', float64[:]),
#        ('vPrev', float64[:]),
#        ('aPrev', float64[:]),
#        ('uTemp', float64[:]),
#        ('vTemp', float64[:]),
#        ('velocity', float64[:]),
#        ('acceleration', float64[:]),
#        ('stiffness', float64[:,:])
#        ]
#
#@jitclass(spec)
class Node:
    def __init__(self, dof, position, initValues, alphaNM, betaNM, timeStep):
        self.dof = dof
        self.referencePosition = array(position)
        self.position = array(position)
        self.displacement = array(initValues)
        self.forces = zeros(dof)
        self.alphaNM = alphaNM
        self.betaNM = betaNM
        self.timeStep = timeStep
        self.sigma = zeros(6)
        self.weightFactor = 0.0
        self.vonMises = 0.0
        self.uPrev = array(initValues)
        self.vPrev = zeros(dof)
        self.aPrev = zeros(dof)
        self.uTemp = array(initValues)
        self.vTemp = zeros(dof)
        self.velocity = zeros(dof)
        self.acceleration = zeros(dof)
        self.stiffness = zeros((dof, dof))
        
    def resetStresses(self):
        self.sigma = zeros(6)
        self.vonMises = 0.0


    def resetWeightFactor(self):
        self.weightFactor = 0.0


    def resetForces(self):
        self.forces = zeros(self.dof)


    def computeVelocitiesAndAccelerations(self):
        # commented lines left for future reference
        # apparently a memory management or stride error occurs here
        # value of uPrev, vPrev, aPrev change here, although untouched
        #print(vPrev)
        #for i in range(0, self.dof):
            #print(i, self.acceleration.shape, self.displacement, self.uTemp.shape)
            #self.uTemp[i] = self.uPrev[i] + self.timeStep * self.vPrev[i] \
            #                + 1.0/2.0 * self.timeStep**2 * (1.0 - 2.0 * self.alphaNM) * self.aPrev[i]
            #self.vTemp[i] = self.vPrev[i] + self.timeStep * (1.0 - self.betaNM) * self.aPrev[i]
            #self.acceleration[i] = (self.displacement[i] - self.uTemp[i]) \
            #                       / (self.alphaNM * self.timeStep**2)
            #self.velocity[i] = self.vTemp[i] + self.betaNM * self.timeStep * self.acceleration[i]
        #self.uTemp = self.uPrev + self.timeStep * self.vPrev + 0.5 * self.timeStep**2 * (1 - 2 * self.alphaNM) * self.aPrev
        #self.vTemp = self.vPrev + self.timeStep * (1 - self.betaNM) * self.aPrev
        # this is vector addition
        self.acceleration = (self.displacement - self.uTemp) / (self.alphaNM * self.timeStep**2)
        self.velocity = self.vTemp + self.betaNM * self.timeStep * self.acceleration
        #print(self.vPrev, "\n")
            
            
    def updatePosition(self):
        # update history variables
        self.uPrev = self.displacement
        self.vPrev = self.velocity
        self.aPrev = self.acceleration
        # update positions
        self.position[0:3] = self.referencePosition[0:3] + self.displacement[0:3]
        self.position[3] = self.referencePosition[3] + self.displacement[3] - 300
        # temporary variables for Newmark calculated here since otherwise error occurs
        self.uTemp = self.uPrev + self.timeStep * self.vPrev + 0.5 * self.timeStep**2 * (1 - 2 * self.alphaNM) * self.aPrev
        self.vTemp = self.vPrev + self.timeStep * (1 - self.betaNM) * self.aPrev

