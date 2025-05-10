# element class 

from node import Node
from numpy import linspace, array, zeros, ones, matmul, insert, delete, round, count_nonzero, float64, log
from numpy.linalg import inv, det
from math import sqrt
from numba import njit
#from time import sleep
#from numba import jit, jitclass, int32, float64, deferred_type, optional
#from numba.types import pyobject
#
#node_type = deferred_type()
#
#spec = [
#        ('numberOfNodes', int32),
#        ('dofPerNode', int32),
#        ('nodes', optional(node_type)),
#        ('gradU', float64[:,:,:]),
#        ('gradUDot', float64[:,:,:]),
#        ('theta0', float64[:]),
#        ('theta', float64[:]),
#        ('thetaDot', float64[:]),
#        ('gradTheta', float64[:,:]),
#        ('vonMises', float64[:]),
#        ('stiffness', float64[:,:]),
#        ('forces', float64[:,:]),
#        ('gaussPoints', float64[:,:]),
#        ('numberOfGaussPoints', int32),
#        ('weights', float64[:]),
#        ('dVol', float64[:]),
#        ('shapeFunctionValues', float64[:,:,:]),
#        ('bmatrix', float64[:,:,:,:]),
#        ('nodePositions', float64[:,:]),
#        ('mC', float64[:,:]),
#        ('lmu', float64),
#        ('llambda', float64),
#        ('alphaS', float64),
#        ('kappa', float64),
#        ('alphaT', float64),
#        ('cp', float64),
#        ('rho', float64),
#        ('rT', float64),
#        ('epsilon', float64[:,:]),
#        ('epsilonDot', float64[:,:]),
#        ('traceEps', float64[:]),
#        ('traceEpsDot', float64[:]),
#        ('sigma', float64[:,:]),
#        ('qlin', float64[:,:]),
#        ('qlinpGradTheta', float64[:,:]),
#        ('sigmapTheta', float64[:,:]),
#        ('balanceOfEnergy', float64[:]),
#        ('balanceOfEnergypTheta', float64[:]),
#        ('balanceOfEnergypThetaDot', float64),
#        ('balanceOfEnergypEpsDot', float64[:,:])
#        ]


# this is calculated externally because numba cannot completely support nested custom classes currently
@njit
def computeStiffness(fv, fa, stiffness, numberOfGaussPoints, dofPerNode, numberOfNodes, bmatrix, mC, dVol, shapeFunctionValues, balanceOfEnergy, balanceOfEnergypTheta, balanceOfEnergypThetaDot, balanceOfEnergypEpsDot, sigmapTheta, qlinpGradTheta):
    #dof = self.dofPerNode      # for brevity's sake
    for gp in range(0, numberOfGaussPoints):
        for node1 in range(0, numberOfNodes):
            
            for node2 in range(0, numberOfNodes):
                
                ## This loop is not gonna change.. bmatrix will have non linear terms also
                ## Mechanical part K_uu
                ## First term in momentum balance law
                for i in range(0, 3): 
                    for j in range(0, 3):
                        for k in range(0, 2*3):
                            for l in range(0, 2*3):
                                stiffness[node1*dofPerNode + i][node2*dofPerNode + j] \
                                                  += bmatrix[gp][node1][k][i] * mC[gp][k][l] \
                                                   * bmatrix[gp][node2][l][j] * dVol[gp]

                # thermal part K_tt
                for i in range(0, 3):
                    for j in range(0, 3):
                        stiffness[node1*dofPerNode + 3][node2*dofPerNode + 3] \
                                          += shapeFunctionValues[gp][node1][i] * qlinpGradTheta[i][j] \
                                           * shapeFunctionValues[gp][node2][j] * dVol[gp]

                stiffness[node1*dofPerNode + 3][node2*dofPerNode + 3] \
                                  += shapeFunctionValues[gp][node1][3] * balanceOfEnergypTheta[gp] \
                                   * shapeFunctionValues[gp][node2][3] * dVol[gp]
                                   
                stiffness[node1*dofPerNode + 3][node2*dofPerNode + 3] \
                                  += shapeFunctionValues[gp][node1][3] * balanceOfEnergypThetaDot \
                                   * shapeFunctionValues[gp][node2][3] * fv * dVol[gp]
  
                ## This loop is not gonna change.. bmatrix will have non linear terms also
                ## coupling part K_ut
                ## Sirst term in momentum balance law
                for i in range(0, 3):
                    for j in range(0, 2*3):
                        stiffness[node1*dofPerNode + i][node2*dofPerNode + 3] \
                                          += bmatrix[gp][node1][j][i] * sigmapTheta[gp][j] \
                                           * shapeFunctionValues[gp][node2][3] * dVol[gp]

                # coupling part K_tu
                for i in range(0, 3):
                    for j in range(0, 2*3):
                        stiffness[node1*dofPerNode + 3][node2*dofPerNode + i] \
                                          += shapeFunctionValues[gp][node1][3] * balanceOfEnergypEpsDot[gp][j] \
                                           * bmatrix[gp][node2][j][i] * fv * dVol[gp]
                                           
# return stiffness
                                           
@njit
def computeForces(forces, numberOfGaussPoints, dofPerNode, numberOfNodes, bmatrix, sigma, dVol, shapeFunctionValues, qlin, balanceOfEnergy):
#    forces = zeros((numberOfNodes, dofPerNode))
    for gp in range(0, numberOfGaussPoints):
        for node in range(0, numberOfNodes):
            #mechanical part f_u
            for i in range(0, 3):
                for j in range(0, 2*3):
                    forces[node][i] -= bmatrix[gp][node][j][i] * sigma[gp][j] * dVol[gp]
                    
            # thermal part f_tt
            for i in range(0, 3):
                forces[node][3] -= shapeFunctionValues[gp][node][i] * qlin[gp][i] * dVol[gp]
            forces[node][3] -= shapeFunctionValues[gp][node][3] * balanceOfEnergy[gp] * dVol[gp]
    
# return forces
            
#@jitclass(spec)
class Element:
    def __init__(self, nodesPerElement, dofPerNode, numberOfGaussPoints, mC, theta0, lmu, llambda, \
                 alphaS, kappa, alphaT, cp, rho, rT): 
        self.numberOfNodes = nodesPerElement
        self.dofPerNode = dofPerNode
        self.nodes = []
        self.gradU = zeros((numberOfGaussPoints, 3, 3), dtype=float64)
        self.gradUDot = zeros((numberOfGaussPoints, 3, 3), dtype=float64)
        self.theta0 = array([theta0] * numberOfGaussPoints, dtype=float64)
        self.theta = zeros(numberOfGaussPoints, dtype=float64)
        self.thetaDot = zeros(numberOfGaussPoints, dtype=float64)
        self.gradTheta = zeros((numberOfGaussPoints, 3), dtype=float64)
        self.vonMises = zeros(numberOfGaussPoints, dtype=float64) 
        self.stiffness = zeros((nodesPerElement * dofPerNode, nodesPerElement * dofPerNode), dtype=float64)
        self.forces = zeros((nodesPerElement, dofPerNode), dtype=float64)
        #self.gaussPoints = []
        self.numberOfGaussPoints = numberOfGaussPoints
        self.weights = ones(nodesPerElement, dtype=float64) # can be customized
        self.dVol = zeros(nodesPerElement, dtype=float64)
        self.shapeFunctionValues = zeros((numberOfGaussPoints, nodesPerElement, dofPerNode), dtype=float64) 
        self.bmatrix = zeros((numberOfGaussPoints, nodesPerElement, 2*3, 3), dtype=float64) 
        #self.nodePositions = []

        # material parameters
        self.lmu = lmu
        self.llambda = llambda
        self.alphaS = alphaS
        self.kappa = kappa
        self.alphaT = alphaT
        self.cp = cp
        self.rho = rho
        self.rT = rT

        # material fields 
        self.mC = mC
        self.epsilon =     zeros((numberOfGaussPoints, 6), dtype=float64)
        self.epsilonDot =  zeros((numberOfGaussPoints, 6), dtype=float64)
        self.traceEps =    zeros((numberOfGaussPoints), dtype=float64)
        self.traceEpsDot = zeros((numberOfGaussPoints), dtype=float64)
        self.sigma =       zeros((numberOfGaussPoints, 6), dtype=float64)
        self.qlin  =       zeros((numberOfGaussPoints, 3), dtype=float64)
        self.qlinpGradTheta = zeros((3, 3), dtype=float64)
        self.sigmapTheta = zeros((numberOfGaussPoints, 6), dtype=float64)
        self.balanceOfEnergy = zeros(numberOfGaussPoints, dtype=float64)
        self.balanceOfEnergypTheta = zeros(numberOfGaussPoints, dtype=float64) 
        self.balanceOfEnergypThetaDot = 0.0
        self.balanceOfEnergypEpsDot = zeros((numberOfGaussPoints, 6), dtype=float64)
        
####################################################################################################        
        # For Project
        # Declaring determinant
        #self.J =     zeros(numberOfGaussPoints, dtype=float64)
        
####################################################################################################
        
        ## Positions of nodes in isoparametric coordinates
        self.nodePositions = array([[-1.0, -1.0, -1.0],       
                                    [ 1.0, -1.0, -1.0],
                                    [ 1.0,  1.0, -1.0],
                                    [-1.0,  1.0, -1.0],
                                    [-1.0, -1.0,  1.0],
                                    [ 1.0, -1.0,  1.0],
                                    [ 1.0,  1.0,  1.0],
                                    [-1.0,  1.0,  1.0]], dtype=float64)

#        gaussPos = 1.0/sqrt(3)
        
        ## Positions of gauss points 
        self.gaussPoints = self.nodePositions / sqrt(3)
        
        
#        = array([[-gaussPos, -gaussPos,  gaussPos],
#                                  [ gaussPos, -gaussPos,  gaussPos],
#                                  [ gaussPos,  gaussPos,  gaussPos],
#                                  [-gaussPos,  gaussPos,  gaussPos],
#                                  [-gaussPos, -gaussPos, -gaussPos],
#                                  [ gaussPos, -gaussPos, -gaussPos],
#                                  [ gaussPos,  gaussPos, -gaussPos],
#                                  [-gaussPos,  gaussPos, -gaussPos]], dtype=float64)
#    
    
#        self.nodePositions.append([-1.0, -1.0,  1.0])
#        self.nodePositions.append([ 1.0, -1.0,  1.0])
#        self.nodePositions.append([ 1.0,  1.0,  1.0])
#        self.nodePositions.append([-1.0,  1.0,  1.0])
#        self.nodePositions.append([-1.0, -1.0, -1.0])
#        self.nodePositions.append([ 1.0, -1.0, -1.0])
#        self.nodePositions.append([ 1.0,  1.0, -1.0])
#        self.nodePositions.append([-1.0,  1.0, -1.0])
#

#        self.gaussPoints.append([-gaussPos, -gaussPos,  gaussPos])
#        self.gaussPoints.append([ gaussPos, -gaussPos,  gaussPos])
#        self.gaussPoints.append([ gaussPos,  gaussPos,  gaussPos])
#        self.gaussPoints.append([-gaussPos,  gaussPos,  gaussPos])
#        self.gaussPoints.append([-gaussPos, -gaussPos, -gaussPos])
#        self.gaussPoints.append([ gaussPos, -gaussPos, -gaussPos])
#        self.gaussPoints.append([ gaussPos,  gaussPos, -gaussPos])
#        self.gaussPoints.append([-gaussPos,  gaussPos, -gaussPos])
#        
#        self.nodePositions.append([-1.0, -1.0, -1.0])
#        self.nodePositions.append([ 1.0, -1.0, -1.0])
#        self.nodePositions.append([ 1.0,  1.0, -1.0])
#        self.nodePositions.append([-1.0,  1.0, -1.0])
#        self.nodePositions.append([-1.0, -1.0,  1.0])
#        self.nodePositions.append([ 1.0, -1.0,  1.0])
#        self.nodePositions.append([ 1.0,  1.0,  1.0])
#        self.nodePositions.append([-1.0,  1.0,  1.0])

#        gaussPos = 1.0/sqrt(3)
#        self.gaussPoints.append([-gaussPos, -gaussPos, -gaussPos])
#        self.gaussPoints.append([ gaussPos, -gaussPos, -gaussPos])
#        self.gaussPoints.append([ gaussPos,  gaussPos, -gaussPos])
#        self.gaussPoints.append([-gaussPos,  gaussPos, -gaussPos])
#        self.gaussPoints.append([-gaussPos, -gaussPos,  gaussPos])
#        self.gaussPoints.append([ gaussPos, -gaussPos,  gaussPos])
#        self.gaussPoints.append([ gaussPos,  gaussPos,  gaussPos])
#        self.gaussPoints.append([-gaussPos,  gaussPos,  gaussPos])
    
#        self.gaussPoints = self.nodePositions / sqrt(3)
        
#        gaussPos = 1.0/sqrt(3)
#        self.gaussPoints.append([-gaussPos, -gaussPos,  gaussPos])
#        self.gaussPoints.append([ gaussPos, -gaussPos,  gaussPos])
#        self.gaussPoints.append([ gaussPos,  gaussPos,  gaussPos])
#        self.gaussPoints.append([-gaussPos,  gaussPos,  gaussPos])
#        self.gaussPoints.append([-gaussPos, -gaussPos, -gaussPos])
#        self.gaussPoints.append([ gaussPos, -gaussPos, -gaussPos])
#        self.gaussPoints.append([ gaussPos,  gaussPos, -gaussPos])
#        self.gaussPoints.append([-gaussPos,  gaussPos, -gaussPos])

    ## Understood == to append the node data in nodes
    def addNode(self, node):                                                             
        if len(self.nodes) < self.numberOfNodes:
            self.nodes.append(node)
        else:
            raise ValueError("Trying to assign more than 8 nodes to element")

    
    ## Understood == shape functions and derivatives
    # position are the isoparametric coordinates of the node, 
    # point are the isoparametric coordinates where you want to evaluate the shape function
    #@jit(nopython=True)#(float64(list(float64)), list(float64))
    def shapeFunction(self, position, point):                                            
        return 1.0/8.0 * (1.0 + position[0]*point[0]) * (1.0 + position[1]*point[1]) * (1.0 + position[2]*point[2])
    
    ## Understood == Derivatives of shape function
    def shapeFunctionX(self, position, point):
        return 1.0/8.0 * position[0] * (1.0 + position[1]*point[1]) * (1.0 + position[2]*point[2])
    
    def shapeFunctionY(self, position, point):
        return 1.0/8.0 * position[1] * (1.0 + position[2]*point[2]) * (1.0 + position[0]*point[0])
    
    def shapeFunctionZ(self, position, point):
        return 1.0/8.0 * position[2] * (1.0 + position[0]*point[0]) * (1.0 + position[1]*point[1])
    
    ## Understood == The sizes of the matrices 
    def resetStresses(self):
        self.mC = zeros((self.numberOfGaussPoints, 6, 6))  ### uncomment if mC isn't constant
        self.epsilon =     zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.epsilonDot =  zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.traceEps =    zeros((self.numberOfGaussPoints), dtype=float64)
        self.traceEpsDot = zeros((self.numberOfGaussPoints), dtype=float64)
        self.sigma =       zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.vonMises =    zeros(self.numberOfGaussPoints, dtype=float64) 
        self.qlin  =       zeros((self.numberOfGaussPoints, 3), dtype=float64)
        self.qlinpGradTheta = zeros((3, 3), dtype=float64)
        self.sigmapTheta = zeros((self.numberOfGaussPoints, 6), dtype=float64)
        self.balanceOfEnergy = zeros(self.numberOfGaussPoints, dtype=float64)
        self.balanceOfEnergypTheta = zeros(self.numberOfGaussPoints, dtype=float64) 
        self.balanceOfEnergypThetaDot = 0.0
        self.balanceOfEnergypEpsDot = zeros((self.numberOfGaussPoints, 6), dtype=float64)
        


    def resetStiffness(self):
        self.stiffness = zeros((self.numberOfNodes * self.dofPerNode, self.numberOfNodes * self.dofPerNode), dtype=float64)
        
        
    def resetForces(self):
        self.forces = zeros((self.numberOfNodes, self.dofPerNode), dtype=float64)

    #@jit
    def calculateShapeFunctions(self):
        # the following variables are only needed locally, so we don't have to save these in the element
        for gp in range(0, self.numberOfGaussPoints):
            isoDeformationGrad = zeros((3, 3), dtype=float64)
            inverseIsoDefoGrad = zeros((3, 3), dtype=float64)
            shapeFunctionDerivatives = zeros((self.numberOfNodes, 3), dtype=float64)
            
            ## Finding derivatives of shape functions at gauss points for all nodes in an element
            for node in range(0, self.numberOfNodes):
                # we don't need to save these in the element for further calculations
                shapeFunctionDerivatives[node] = array([self.shapeFunctionX(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionY(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionZ(self.nodePositions[node], self.gaussPoints[gp])])
    
            ## Nodes is 8*3 and shapeFunctionDerivatives is 8*3   position and derivatives
            for i in range(0,3):
                for j in range(0,3):
                    for k in range(0, self.numberOfNodes):
                        isoDeformationGrad[i][j] += self.nodes[k].position[i] * shapeFunctionDerivatives[k][j]   ###refPos?

            # Get value of multiplication of jacobian and weights
            self.dVol[gp] = det(isoDeformationGrad) * self.weights[gp]

            # Get inverse Jacobi matrix
            inverseIsoDefoGrad = inv(isoDeformationGrad) 

            for k in range(0, self.numberOfNodes):
                # this array saves all nodal shape function values at each Gauss point
                self.shapeFunctionValues[gp][k] = array([inverseIsoDefoGrad[0][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[0][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[0][2] * shapeFunctionDerivatives[k][2], \
                                                         inverseIsoDefoGrad[1][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[1][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[1][2] * shapeFunctionDerivatives[k][2], \
                                                         inverseIsoDefoGrad[2][0] * shapeFunctionDerivatives[k][0] \
                                                       + inverseIsoDefoGrad[2][1] * shapeFunctionDerivatives[k][1] \
                                                       + inverseIsoDefoGrad[2][2] * shapeFunctionDerivatives[k][2], \
                                                         self.shapeFunction(self.nodePositions[k], self.gaussPoints[gp])])
    
    
#                self.bmatrix[gp][k] = array([[self.shapeFunctionValues[gp][k][0], 0, 0],\
#                                             [0, self.shapeFunctionValues[gp][k][1], 0],\
#                                             [0, 0, self.shapeFunctionValues[gp][k][2]],\
#                                             [self.shapeFunctionValues[gp][k][1], self.shapeFunctionValues[gp][k][0], 0],\
#                                             [0, self.shapeFunctionValues[gp][k][2], self.shapeFunctionValues[gp][k][1]],\
#                                             [self.shapeFunctionValues[gp][k][2], 0, self.shapeFunctionValues[gp][k][0]]])
    
   
    
    def computeVelocitiesAndAccelerations(self):
        for node in self.nodes:
            node.computeUVA()


    # the field variables for Newton-Raphson
    #@jit
    def computeFieldVars(self):
        self.gradU     = zeros((self.numberOfGaussPoints, 3, 3), dtype=float64)
        self.gradUDot  = zeros((self.numberOfGaussPoints, 3, 3), dtype=float64)
        self.theta    = zeros(self.numberOfGaussPoints, dtype=float64)
        self.thetaDot  = zeros(self.numberOfGaussPoints, dtype=float64)
        self.gradTheta = zeros((self.numberOfGaussPoints, 3), dtype=float64)         
        
        for gp in range(0, self.numberOfGaussPoints):
            for k in range(0, self.numberOfNodes):
                for i in range(0, 3):
                    for j in range(0, 3):
                        self.gradU[gp][i][j]    += self.shapeFunctionValues[gp][k][j] * self.nodes[k].displacement[i]
                        self.gradUDot[gp][i][j] += self.shapeFunctionValues[gp][k][j] * self.nodes[k].velocity[i]
                        
                    self.gradTheta[gp][i] += self.shapeFunctionValues[gp][k][i] * self.nodes[k].displacement[3]
                    
                self.theta[gp] += self.shapeFunctionValues[gp][k][3] * self.nodes[k].displacement[3]
                self.thetaDot[gp] += self.shapeFunctionValues[gp][k][3] * self.nodes[k].velocity[3]
                #print(self.nodes[k].displacement)
            #print(self.gradU[gp])


    # "material routine"
    def computeStresses(self):
        for gp in range(0, self.numberOfGaussPoints):
            cauchy                    = zeros((3, 3), dtype=float64)
            invCauchy                 = zeros((3, 3), dtype=float64)
            DeformationGrad           = zeros((3, 3), dtype=float64)
            ShapeFunctionDerivatives  = zeros((self.numberOfNodes, 3), dtype=float64)
            F                         = zeros((3, 3), dtype=float64)
            
######################################################################################################################################################################            
            # linear strain vector in Voigt notation
            # Calculations are based on E - green lagrange strain tensor. E = 0.5*(transpose(F).F)
            # Expanding E gives E = 0.5*(transpose(GradU) + GradU) = epsilon
            #self.epsilon[gp][0] = self.gradU[gp][0][0]
            #self.epsilon[gp][1] = self.gradU[gp][1][1]
            #self.epsilon[gp][2] = self.gradU[gp][2][2]
            #self.epsilon[gp][3] = self.gradU[gp][0][1] + self.gradU[gp][1][0]
            #self.epsilon[gp][4] = self.gradU[gp][1][2] + self.gradU[gp][2][1]
            #self.epsilon[gp][5] = self.gradU[gp][2][0] + self.gradU[gp][0][2]
            
         
            # Implemented for Project
            # Calculation of non linear E in Voigt notation
            # Loop for implementation of linear entries    
            self.epsilon[gp][0] = self.gradU[gp][0][0]
            self.epsilon[gp][1] = self.gradU[gp][1][1]
            self.epsilon[gp][2] = self.gradU[gp][2][2]
            self.epsilon[gp][3] = self.gradU[gp][0][1] + self.gradU[gp][1][0]
            self.epsilon[gp][4] = self.gradU[gp][1][2] + self.gradU[gp][2][1]
            self.epsilon[gp][5] = self.gradU[gp][2][0] + self.gradU[gp][0][2]
            # Loop to add all non linear entrieds into E
            for j in range(0,3):
                self.epsilon[gp][0] += 0.5 * self.gradU[gp][j][0] * self.gradU[gp][j][0]
                self.epsilon[gp][1] += 0.5 * self.gradU[gp][j][1] * self.gradU[gp][j][1]
                self.epsilon[gp][2] += 0.5 * self.gradU[gp][j][2] * self.gradU[gp][j][2]
                self.epsilon[gp][3] += self.gradU[gp][j][0] * self.gradU[gp][j][1]
                self.epsilon[gp][4] += self.gradU[gp][j][1] * self.gradU[gp][j][2]
                self.epsilon[gp][5] += self.gradU[gp][j][0] * self.gradU[gp][j][2]
                            
#######################################################################################################################################################################

            # Calculation of time derivative of epsilon in Voigt notation
            #self.epsilonDot[gp][0] = self.gradUDot[gp][0][0]
            #self.epsilonDot[gp][1] = self.gradUDot[gp][1][1]
            #self.epsilonDot[gp][2] = self.gradUDot[gp][2][2]
            #self.epsilonDot[gp][3] = self.gradUDot[gp][0][1] + self.gradUDot[gp][1][0]
            #self.epsilonDot[gp][4] = self.gradUDot[gp][1][2] + self.gradUDot[gp][2][1]
            #self.epsilonDot[gp][5] = self.gradUDot[gp][2][0] + self.gradUDot[gp][0][2]
            
            # Implemented for Project
            # Calculation of time derivative of epsilonDot in Voigt notation
            # Loop for implementation of linear entries    
            self.epsilonDot[gp][0] = self.gradUDot[gp][0][0]
            self.epsilonDot[gp][1] = self.gradUDot[gp][1][1]
            self.epsilonDot[gp][2] = self.gradUDot[gp][2][2]
            self.epsilonDot[gp][3] = self.gradUDot[gp][0][1] + self.gradUDot[gp][1][0]
            self.epsilonDot[gp][4] = self.gradUDot[gp][1][2] + self.gradUDot[gp][2][1]
            self.epsilonDot[gp][5] = self.gradUDot[gp][2][0] + self.gradUDot[gp][0][2]
            # Loop to add all non linear entrieds into E
            for j in range(0,3):
                self.epsilonDot[gp][0] += 0.5 * self.gradUDot[gp][j][0] * self.gradUDot[gp][j][0]
                self.epsilonDot[gp][1] += 0.5 * self.gradUDot[gp][j][1] * self.gradUDot[gp][j][1]
                self.epsilonDot[gp][2] += 0.5 * self.gradUDot[gp][j][2] * self.gradUDot[gp][j][2]
                self.epsilonDot[gp][3] += self.gradUDot[gp][j][0] * self.gradUDot[gp][j][1]
                self.epsilonDot[gp][4] += self.gradUDot[gp][j][1] * self.gradUDot[gp][j][2]
                self.epsilonDot[gp][5] += self.gradUDot[gp][j][0] * self.gradUDot[gp][j][2]
                             
                   
            # Calculation of trace(epsilon)
            for i in range(0,3):
                self.traceEps[gp] += self.epsilon[gp][i]
                self.traceEpsDot[gp] += self.epsilonDot[gp][i]
                
#######################################################################################################################################################################                
                
            # Implemented for Project    
            # Step 1. Conversion of E to C
            cauchy[0][0] = 2.0 * self.epsilon[gp][0] + 1.0
            cauchy[0][1] = self.epsilon[gp][3]
            cauchy[0][2] = self.epsilon[gp][5]
            cauchy[1][0] = self.epsilon[gp][3]
            cauchy[1][1] = 2.0 * self.epsilon[gp][1] + 1.0
            cauchy[1][2] = self.epsilon[gp][4]
            cauchy[2][0] = self.epsilon[gp][5]
            cauchy[2][1] = self.epsilon[gp][4]
            cauchy[2][2] = 2.0 * self.epsilon[gp][2] + 1.0
            
            # Step 2. Calculate the inverse of cauchy
            invCauchy = inv(cauchy)
            
            # Step. 3 Calculate J = det(F)            
            # Finding derivatives of shape functions at gauss points for all nodes in an element
            for node in range(0, self.numberOfNodes):
                # we don't need to save these in the element for further calculations
                ShapeFunctionDerivatives[node] = array([self.shapeFunctionX(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionY(self.nodePositions[node], self.gaussPoints[gp]), \
                                                        self.shapeFunctionZ(self.nodePositions[node], self.gaussPoints[gp])])
            for i in range(0,3):
                for j in range(0,3):
                    for k in range(0, self.numberOfNodes):
                        DeformationGrad[i][j] += self.nodes[k].position[i] * ShapeFunctionDerivatives[k][j]
                        
            # Step. 4 Calculation determinant of F: F = gradU + I 
            for i in range(0, 3):
                for j in range(0, 3):
                    if i == j:
                        F[i][j] =  self.gradU[gp][i][j] + 1.0
                    else:
                        F[i][j] =  self.gradU[gp][i][j]
                    
            J = det(F)
            
            
#######################################################################################################################################################################                        
            
            # Implemented for Project
            # Calculation of Strain-Displacement Matrix
            for k in range(0, self.numberOfNodes):
                bmatrixl = zeros((2*3, 3), dtype=float64)    
                bmatrixl         =     array([[self.shapeFunctionValues[gp][k][0], 0, 0],\
                                              [0, self.shapeFunctionValues[gp][k][1], 0],\
                                              [0, 0, self.shapeFunctionValues[gp][k][2]],\
                                              [self.shapeFunctionValues[gp][k][1], self.shapeFunctionValues[gp][k][0], 0],\
                                              [0, self.shapeFunctionValues[gp][k][2], self.shapeFunctionValues[gp][k][1]],\
                                              [self.shapeFunctionValues[gp][k][2], 0, self.shapeFunctionValues[gp][k][0]]])
    
                # bmatrix non linear
                bmatrixnl = zeros((2*3, 3), dtype=float64) 
                bmatrixnl        =     array([[self.gradU[gp][0][0] * self.shapeFunctionValues[gp][k][0],\
                                               self.gradU[gp][1][0] * self.shapeFunctionValues[gp][k][0],\
                                               self.gradU[gp][2][0] * self.shapeFunctionValues[gp][k][0]],\
                                              [self.gradU[gp][0][1] * self.shapeFunctionValues[gp][k][1],\
                                               self.gradU[gp][1][1] * self.shapeFunctionValues[gp][k][1],\
                                               self.gradU[gp][2][1] * self.shapeFunctionValues[gp][k][1]],\
                                              [self.gradU[gp][0][2] * self.shapeFunctionValues[gp][k][2],\
                                               self.gradU[gp][1][2] * self.shapeFunctionValues[gp][k][2],\
                                               self.gradU[gp][2][2] * self.shapeFunctionValues[gp][k][2]],\
                                              [self.gradU[gp][0][1] * self.shapeFunctionValues[gp][k][0] + self.gradU[gp][0][0] * self.shapeFunctionValues[gp][k][1],\
                                               self.gradU[gp][1][1] * self.shapeFunctionValues[gp][k][0] + self.gradU[gp][1][0] * self.shapeFunctionValues[gp][k][1],\
                                               self.gradU[gp][2][1] * self.shapeFunctionValues[gp][k][0] + self.gradU[gp][2][0] * self.shapeFunctionValues[gp][k][1]],\
                                              [self.gradU[gp][0][2] * self.shapeFunctionValues[gp][k][1] + self.gradU[gp][0][1] * self.shapeFunctionValues[gp][k][2],\
                                               self.gradU[gp][1][2] * self.shapeFunctionValues[gp][k][1] + self.gradU[gp][1][1] * self.shapeFunctionValues[gp][k][2],\
                                               self.gradU[gp][2][2] * self.shapeFunctionValues[gp][k][1] + self.gradU[gp][2][1] * self.shapeFunctionValues[gp][k][2]],\
                                              [self.gradU[gp][0][2] * self.shapeFunctionValues[gp][k][0] + self.gradU[gp][0][0] * self.shapeFunctionValues[gp][k][2],\
                                               self.gradU[gp][1][2] * self.shapeFunctionValues[gp][k][0] + self.gradU[gp][1][0] * self.shapeFunctionValues[gp][k][2],\
                                               self.gradU[gp][2][2] * self.shapeFunctionValues[gp][k][0] + self.gradU[gp][2][0] * self.shapeFunctionValues[gp][k][2]]])
                
    
                for i in range(0, 2*3):
                    for j in range(0, 3):
                        self.bmatrix[gp][k][i][j] = bmatrixl[i][j] + bmatrixnl[i][j]
                        
#######################################################################################################################################################################             
            
            # Calculation linearise second piola stress which is reqiuired in the calculation of stiffness matrix
            # Implementation of linear S 
            #self.sigma[gp][0] = 2.0 * self.lmu * self.epsilon[gp][0] + self.llambda * self.traceEps[gp] \
            #                  - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp])
            #self.sigma[gp][1] = 2.0 * self.lmu * self.epsilon[gp][1] + self.llambda * self.traceEps[gp] \
            #                  - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp])
            #self.sigma[gp][2] = 2.0 * self.lmu * self.epsilon[gp][2] + self.llambda * self.traceEps[gp] \
            #                  - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp])
            #self.sigma[gp][3] = self.lmu * self.epsilon[gp][3]
            #self.sigma[gp][4] = self.lmu * self.epsilon[gp][4]
            #self.sigma[gp][5] = self.lmu * self.epsilon[gp][5]
            
            
            # Implemented for Project
            # Implementation of non linear S  
            self.sigma[gp][0] = self.lmu * (1 - invCauchy[0][0]) + self.llambda * log(J) * invCauchy[0][0] \
                            - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp]) * invCauchy[0][0]
            self.sigma[gp][1] = self.lmu * (1 - invCauchy[1][1]) + self.llambda * log(J) * invCauchy[1][1] \
                            - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp]) * invCauchy[1][1]
            self.sigma[gp][2] = self.lmu * (1 - invCauchy[2][2]) + self.llambda * log(J) * invCauchy[2][2] \
                            - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp]) * invCauchy[2][2]
            self.sigma[gp][3] = - self.lmu * 2.0 * invCauchy[0][1] + self.llambda * log(J) * 2.0 * invCauchy[0][1] \
                            - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp]) * 2.0 * invCauchy[0][1]
            self.sigma[gp][4] = - self.lmu * 2.0 * invCauchy[1][2] + self.llambda * log(J) * 2.0 * invCauchy[1][2] \
                            - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp]) * 2.0 * invCauchy[1][2]
            self.sigma[gp][5] = - self.lmu * 2.0 * invCauchy[0][2] + self.llambda * log(J) * 2.0 * invCauchy[0][2] \
                            - 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp]) * 2.0 * invCauchy[0][2]
            
#######################################################################################################################################################################            
            
            # Implemented for Project
            # Implementation of qlin 
            # Consideration of linear theory - Heat flux is deformation independent
            self.qlin[gp][0] = -self.alphaT * self.gradTheta[gp][0]
            self.qlin[gp][1] = -self.alphaT * self.gradTheta[gp][1]
            self.qlin[gp][2] = -self.alphaT * self.gradTheta[gp][2]
            
            
            # Implemented for Project
            # Calculation of  q(heat flux)
            # Implementation of non linear q - future reference
            #self.qlin[gp][0] = -self.alphaT * J * (self.gradTheta[gp][0] * invCauchy[0][0] + self.gradTheta[gp][1] * invCauchy[0][1] + self.gradTheta[gp][2] * invCauchy[0][2])
            #self.qlin[gp][1] = -self.alphaT * J * (self.gradTheta[gp][0] * invCauchy[1][0] + self.gradTheta[gp][1] * invCauchy[1][1] + self.gradTheta[gp][2] * invCauchy[1][2])
            #self.qlin[gp][2] = -self.alphaT * J * (self.gradTheta[gp][0] * invCauchy[2][0] + self.gradTheta[gp][1] * invCauchy[2][1] + self.gradTheta[gp][2] * invCauchy[2][2])
            
#######################################################################################################################################################################           

            self.balanceOfEnergy[gp] = -self.rho * self.cp * self.thetaDot[gp] \
                                       - 3.0 * self.alphaS * self.theta[gp] * self.kappa * self.traceEpsDot[gp] \
                                       + self.rho * self.rT
                                                   
#######################################################################################################################################################################
            
            # elastic tangent modulus in Voigt notation for linear case                                 
            #self.mC[0][0] = 2 * self.lmu + self.llambda
            #self.mC[0][1] = self.llambda
            #self.mC[0][2] = self.llambda
            #self.mC[1][0] = self.llambda
            #self.mC[1][1] = 2 * self.lmu + self.llambda
            #self.mC[1][2] = self.llambda
            #self.mC[2][0] = self.llambda
            #self.mC[2][1] = self.llambda
            #self.mC[2][2] = 2 * self.lmu + self.llambda
            #self.mC[3][3] = self.lmu
            #self.mC[4][4] = self.lmu
            #self.mC[5][5] = self.lmu

#######################################################################################################################################################################
           
            # Implemented for Project
            # elastic tangent modulus in Voigt notation
            # Here mC = ( lmu - llambda * ln(J) + 3.0 * alphaS * kappa * (theta - theta0)) * (C^-1 x C^-1)^T + llambda * (C^-1 x C^-1)
            invCinvCT = zeros((3, 3, 3, 3), dtype=float64)
            invCinvC = zeros((3, 3, 3, 3), dtype=float64)
            mCconst1 = self.lmu - self.llambda * log(J) + 3.0 * self.alphaS * self.kappa * (self.theta[gp] - self.theta0[gp])
            mCconst2 = self.llambda
            
            # Formulation of (C^-1 x C^-1)^T and (C^-1 x C^-1)
            index = [[0,0], [1,1], [2,2], [0,1], [1,2], [0,2]]
            
            for l in range (0,3):               # l = p
                for k in range (0,3):           # k = q
                    for m in range (0,3):       # m = r
                        for n in range (0,3):   # n = s
                            invCinvCT[l][k][m][n] = invCauchy[l][m] * invCauchy[n][k] + invCauchy[l][n] * invCauchy[m][k]
                            invCinvC[l][k][m][n] = invCauchy[l][k] * invCauchy[m][n]
            
            # Formulation of mC which is changing at every gauss point of the element
            for i in range (0,6):
                for j in range (0,6):
                    self.mC[gp][i][j] = mCconst1 * invCinvCT[index[i][0]][index[i][1]][index[j][0]][index[j][1]] \
                                        + mCconst2 * invCinvC[index[i][0]][index[i][1]][index[j][0]][index[j][1]]
        
#######################################################################################################################################################################
            
            # Implemented for Project
            # Calculation of dS/dTheta ( S non linear)                          
            self.sigmapTheta[gp][0] = -3.0 * self.alphaS * self.kappa * invCauchy[0][0]
            self.sigmapTheta[gp][1] = -3.0 * self.alphaS * self.kappa * invCauchy[1][1]
            self.sigmapTheta[gp][2] = -3.0 * self.alphaS * self.kappa * invCauchy[2][2]
            self.sigmapTheta[gp][3] = -3.0 * self.alphaS * self.kappa * 2.0 * invCauchy[0][1]
            self.sigmapTheta[gp][4] = -3.0 * self.alphaS * self.kappa * 2.0 * invCauchy[1][2]
            self.sigmapTheta[gp][5] = -3.0 * self.alphaS * self.kappa * 2.0 * invCauchy[0][2]
  
#######################################################################################################################################################################       
           
            # Implemented for Project
            # Derivative of heat flux with respect to theta
            # For linear theory
            self.qlinpGradTheta[0][0] = -self.alphaT 
            self.qlinpGradTheta[1][1] = -self.alphaT
            self.qlinpGradTheta[2][2] = -self.alphaT
            
            
            # Derivative of heat flux with respect to theta
            # For non linear theory - For future reference
            #self.qlinpGradTheta[0][0] = -self.alphaT * J * invCauchy[0][0]
            #self.qlinpGradTheta[0][1] = -self.alphaT * J * invCauchy[0][1] 
            #self.qlinpGradTheta[0][2] = -self.alphaT * J * invCauchy[0][2] 
            #self.qlinpGradTheta[1][0] = -self.alphaT * J * invCauchy[1][0] 
            #self.qlinpGradTheta[1][1] = -self.alphaT * J * invCauchy[1][1]
            #self.qlinpGradTheta[1][2] = -self.alphaT * J * invCauchy[1][2]
            #self.qlinpGradTheta[2][0] = -self.alphaT * J * invCauchy[2][0]  
            #self.qlinpGradTheta[2][1] = -self.alphaT * J * invCauchy[2][1]
            #self.qlinpGradTheta[2][2] = -self.alphaT * J * invCauchy[2][2] 
            
           
#######################################################################################################################################################################            
            
            # Taking derivative of coupling term in the balance of energy term
            # with respect to Theta
            self.balanceOfEnergypTheta[gp] = -3.0 * self.alphaS * self.kappa * self.traceEpsDot[gp]
            
            # with respect to ThetaDot
            self.balanceOfEnergypThetaDot = -self.rho * self.cp
            
            # with respect to epsilonDOt
            self.balanceOfEnergypEpsDot[gp][0] = -3.0 * self.alphaS * self.kappa * self.theta[gp]
            self.balanceOfEnergypEpsDot[gp][1] = -3.0 * self.alphaS * self.kappa * self.theta[gp]
            self.balanceOfEnergypEpsDot[gp][2] = -3.0 * self.alphaS * self.kappa * self.theta[gp]
            self.balanceOfEnergypEpsDot[gp][3] = 0.0
            self.balanceOfEnergypEpsDot[gp][4] = 0.0
            self.balanceOfEnergypEpsDot[gp][5] = 0.0

#######################################################################################################################################################################

            # von Mises stress
            self.vonMises[gp] += (self.sigma[gp][0] - self.sigma[gp][1])**2 + (self.sigma[gp][1] - self.sigma[gp][2])**2 \
                         + (self.sigma[gp][2] - self.sigma[gp][0])**2 \
                         + 6.0 * (self.sigma[gp][3]**2 + self.sigma[gp][4]**2 + self.sigma[gp][5]**2)
            self.vonMises[gp] = 1.0/2.0 * sqrt(self.vonMises[gp])

            # update nodal stresses and von Mises stresses
            for k in range(0, self.numberOfNodes):
                self.nodes[k].weightFactor += self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                self.nodes[k].vonMises += self.vonMises[gp] * self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                for i in range(0, 2*3):
                    self.nodes[k].sigma[i] += self.sigma[gp][i] * self.shapeFunctionValues[gp][k][3]**2 * self.dVol[gp]
                    
                    
#######################################################################################################################################################################                    

    def computeUVA(self):
        for node in self.nodes:
            node.computeUVA()

    #@jit
    def computeStiffness(self):
        #computeStiffness(fv, fa, self.stiffness, self.numberOfGaussPoints, self.dofPerNode, self.numberOfNodes, self.bmatrix, self.mC, self.dVol, self.shapeFunctionValues, self.balanceOfEnergy, self.balanceOfEnergypTheta, self.balanceOfEnergypThetaDot, self.balanceOfEnergypEpsDot, self.sigmapTheta, self.qlinpGradTheta)
#        # shorthand for Newmark parameter combinations, this is dirty
        fv = self.nodes[0].betaNM/(self.nodes[0].alphaNM * self.nodes[0].timeStep)
        fa = 1.0 / (self.nodes[0].alphaNM * self.nodes[0].timeStep**2)
        
        computeStiffness(fv, fa, self.stiffness, self.numberOfGaussPoints, self.dofPerNode, self.numberOfNodes, self.bmatrix, self.mC, self.dVol, self.shapeFunctionValues, self.balanceOfEnergy, self.balanceOfEnergypTheta, self.balanceOfEnergypThetaDot, self.balanceOfEnergypEpsDot, self.sigmapTheta, self.qlinpGradTheta)
#        #print(fv, fa)
#
#        dof = self.dofPerNode      # for brevity's sake
#        for gp in range(0, self.numberOfGaussPoints):
#            for node1 in range(0, self.numberOfNodes):
#                #mechanical part K_uu
#                for node2 in range(0, self.numberOfNodes):
#                    for i in range(0, 3): 
#                        for j in range(0, 3):
#                            for k in range(0, 2*3):
#                                for l in range(0, 2*3):
#                                    self.stiffness[node1*self.dofPerNode + i][node2*self.dofPerNode + j] \
#                                                      += self.bmatrix[gp][node1][k][i] * self.mC[k][l] \
#                                                       * self.bmatrix[gp][node2][l][j] * self.dVol[gp]
#
#                    # thermal part K_tt
#                    for i in range(0, 3):
#                        for j in range(0, 3):
#                            self.stiffness[node1*self.dofPerNode + 3][node2*self.dofPerNode + 3] \
#                                              += self.shapeFunctionValues[gp][node1][i] * self.qlinpGradTheta[i][j] \
#                                               * self.shapeFunctionValues[gp][node2][j] * self.dVol[gp]
#    
#                    self.stiffness[node1*self.dofPerNode + 3][node2*self.dofPerNode + 3] \
#                                      += self.shapeFunctionValues[gp][node1][3] * self.balanceOfEnergypTheta[gp] \
#                                       * self.shapeFunctionValues[gp][node2][3] * self.dVol[gp]
#                                       
#                    self.stiffness[node1*self.dofPerNode + 3][node2*self.dofPerNode + 3] \
#                                      += self.shapeFunctionValues[gp][node1][3] * self.balanceOfEnergypThetaDot \
#                                       * self.shapeFunctionValues[gp][node2][3] * fv * self.dVol[gp]
#  
#                    # coupling part K_ut
#                    for i in range(0, 3):
#                        for j in range(0, 2*3):a
#                            self.stiffness[node1*self.dofPerNode + i][node2*self.dofPerNode + 3] \
#                                              += self.bmatrix[gp][node1][j][i] * self.sigmapTheta[gp][j] \
#                                               * self.shapeFunctionValues[gp][node2][3] * self.dVol[gp]
#    
#                    # coupling part K_tu
#                    for i in range(0, 3):
#                        for j in range(0, 2*3):
#                            self.stiffness[node1*self.dofPerNode + 3][node2*self.dofPerNode + i] \
#                                              += self.shapeFunctionValues[gp][node1][3] * self.balanceOfEnergypEpsDot[gp][j] \
#                                               * self.bmatrix[gp][node2][j][i] * fv * self.dVol[gp]
#                                               
#    #func = njit(computeStiffness)

    #@jit(nopython=True)
    def computeForces(self):
        computeForces(self.forces, self.numberOfGaussPoints, self.dofPerNode, self.numberOfNodes, self.bmatrix, self.sigma, self.dVol, self.shapeFunctionValues, self.qlin, self.balanceOfEnergy)
#        for gp in range(0, self.numberOfGaussPoints):
#            for node in range(0, self.numberOfNodes):
#                #mechanical part f_u
#                for i in range(0, 3):
#                    for j in range(0, 2*3):
#                        self.forces[node][i] -= self.bmatrix[gp][node][j][i] * self.sigma[gp][j] * self.dVol[gp]
#                        
#                # thermal part f_tt
#                for i in range(0, 3):
#                    self.forces[node][3] -= self.shapeFunctionValues[gp][node][i] * self.qlin[gp][i] * self.dVol[gp]
#                self.forces[node][3] -= self.shapeFunctionValues[gp][node][3] * self.balanceOfEnergy[gp] * self.dVol[gp]
#                
#    
        # move forces to nodes
        for node in range(0, self.numberOfNodes):
            self.nodes[node].forces += self.forces[node]


    def printNodes(self):
        for node in self.nodes:
            node.printCoordinates()

#temp = Node(4, [0.0, 0.0, 0.0, 300.0], [0.0, 0.0, 0.0, 300.0], 1/4, 1/2, 1)
#node_type.define(type(temp))