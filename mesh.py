# mesh class
# Only for cuboids!

from numpy import flip, linspace
from node import *
from element import *
from math import sqrt
from scipy.sparse.linalg import cg

class Cuboid:
    def __init__(self, numberOfElements, numberOfNodes, nodesPerElement, dofPerNode, numberOfGaussPoints, fixedDofs, \
                 length, width, height, initValues, theta0, alphaNM, betaNM, timeStep, \
                 lmu, llambda, alphaS, kappa, alphaT, cp, rho, rT):
        # assuming equal division, can be customized
        nodesPerDim = numberOfNodes / 9    ## calculates number of nodes in one dimension to calculate the 3D coordinates 
        self.nodes = []
        self.elements = []
        self.stiffness = zeros((dofPerNode * numberOfNodes, dofPerNode * numberOfNodes))   
        self.forces    = zeros((dofPerNode * numberOfNodes))
        self.fixedDofs = fixedDofs
        self.dels = 0

        # create and assign nodes
        for z in linspace(0, height, nodesPerDim):
            for y in linspace(0, width, nodesPerDim):
                for x in linspace(0, length, nodesPerDim):
                    self.nodes.append(Node(dofPerNode, [x, y, z, theta0], initValues, alphaNM, betaNM, timeStep))

###################################################################################################################
        # elastic tangent modulus        
        # in general, you'd define this per element
        mC = zeros((numberOfGaussPoints, 6, 6), dtype=float64)
        # defined once here, since it is constant (for linear cases)
        #mC = zeros((6, 6))
        #mC[0][0] = 2 * lmu + llambda
        #mC[0][1] = llambda
        #mC[0][2] = llambda
        #mC[1][0] = llambda
        #mC[1][1] = 2 * lmu + llambda
        #mC[1][2] = llambda
        #mC[2][0] = llambda
        #mC[2][1] = llambda
        #mC[2][2] = 2 * lmu + llambda 
        #mC[3][3] = lmu
        #mC[4][4] = lmu
        #mC[5][5] = lmu

###################################################################################################################
        
        # add empty elements to element list
        for elementNumber in range(0, numberOfElements):
            self.elements.append(Element(nodesPerElement, dofPerNode, numberOfGaussPoints, mC, initValues[3], \
                                 lmu, llambda, alphaS, kappa, alphaT, cp, rho, rT))

        # contains local to global node mapping
        self.elementGeometry = zeros((numberOfElements,nodesPerElement), dtype=int)
                    
        ## ID Matrix
        self.elementGeometry[0] = array([ 0,  1,  4,  3,  9, 10, 13, 12])
        self.elementGeometry[1] = array([ 1,  2,  5,  4, 10, 11, 14, 13])
        self.elementGeometry[2] = array([ 4,  5,  8,  7, 13, 14, 17, 16])        
        self.elementGeometry[3] = array([ 3,  4,  7,  6, 12, 13, 16, 15])        
        self.elementGeometry[4] = array([ 9, 10, 13, 12, 18, 19, 22, 21])        
        self.elementGeometry[5] = array([10, 11, 14, 13, 19, 20, 23, 22])        
        self.elementGeometry[6] = array([13, 14, 17, 16, 22, 23, 26, 25])
        self.elementGeometry[7] = array([12, 13, 16, 15, 21, 22, 25, 24])
                                    
        # assign nodes to elements
        for element in range(0, numberOfElements):
            for node in self.elementGeometry[element]:
                self.elements[element].addNode(self.nodes[node])       ## Look into class Element addNode function
 
        self.dofPerNode = dofPerNode
        self.numberOfNodes = len(self.nodes)
        self.numberOfElements = len(self.elements)


    def setExternalForces(self, externalForces):                       ## have a look at the external forces in input file
        for node in externalForces:
            for dof in range(0, self.dofPerNode):
                self.forces[node*self.dofPerNode + dof] += externalForces[node][dof]


    def resetGlobalStiffness(self):
        self.stiffness = zeros((self.dofPerNode * self.numberOfNodes, self.dofPerNode * self.numberOfNodes))


    def resetGlobalForces(self):
        self.forces = zeros(self.dofPerNode * self.numberOfNodes)


    def resetStresses(self):
        for node in self.nodes:
            node.resetStresses() # includes von Mises stresses
            
        for element in self.elements:
            element.resetStresses()


    def resetWeightFactors(self):
        for node in self.nodes:
            node.resetWeightFactor()


    def resetLocalStiffness(self):
        for element in self.elements:
            element.resetStiffness()


    def resetLocalForces(self):                                        
        for node in self.nodes:
            node.resetForces()
        
        for element in self.elements:
            element.resetForces()


    def computeShapeFunctions(self):
        for element in self.elements:
            element.calculateShapeFunctions()


    def computeVelocitiesAndAccelerations(self):
        for node in self.nodes:
            node.computeVelocitiesAndAccelerations()


    def computeFieldVars(self):
        for element in self.elements:
            element.computeFieldVars()


    # call "material routine" of every element
    def computeStresses(self):        
        for element in self.elements:
            element.computeStresses()


    def computeLocalStiffness(self):
        for element in self.elements:
            element.computeStiffness()


    def computeLocalForces(self):
        for element in self.elements:
            element.computeForces()


    def computeGlobalStiffness(self):
        for element in range(0, self.numberOfElements):
            i = 0
            for node1 in self.elementGeometry[element]:
                j = 0
                for node2 in self.elementGeometry[element]:
                    for dof1 in range(0, self.dofPerNode):
                        for dof2 in range(0, self.dofPerNode):
                            self.stiffness[node1 * self.dofPerNode + dof1][node2 * self.dofPerNode + dof2] \
                                              += self.elements[element].stiffness[i + dof1][j + dof2]
                    j += self.dofPerNode
                i += self.dofPerNode
    


    def computeGlobalForces(self):
        for node in range(0, self.numberOfNodes):
            for dof in range(0, self.dofPerNode):
                self.forces[node*self.dofPerNode + dof] += self.nodes[node].forces[dof]


    def applyBoundaryConditions(self):
        # rows and columns referring to fixed dofs are simply deleted
        self.dels = 0
        for node in self.fixedDofs:
            for dof in self.fixedDofs[node]:
                self.stiffness = delete(self.stiffness, (node*self.dofPerNode + dof - self.dels), axis=0)
                self.stiffness = delete(self.stiffness, (node*self.dofPerNode + dof - self.dels), axis=1)
                self.forces    = delete(self.forces,    (node*self.dofPerNode + dof - self.dels), axis=0)
                self.dels += 1
        

    def computeDisplacements(self):
        #print(self.stiffness)
        #print(self.forces)
        #print(displacements)
        displacements = matmul(inv(self.stiffness), self.forces)
        
        #displacements = cg(matmul(self.stiffness,self.stiffness), matmul(self.stiffness,self.forces))[0]
        #print(displacements[0])
        for node in flip(list(self.fixedDofs.keys())): # has to be traversed in reverse order
            for dof in flip(self.fixedDofs[node]):
                # reintroduce deleted rows for easier assignment
                self.dels -= 1
                displacements  = insert(displacements, node*self.dofPerNode + dof - self.dels, 0.0, axis=0)

        for node in range(0, self.numberOfNodes):
            self.nodes[node].displacement += displacements[node * self.dofPerNode:node * self.dofPerNode + self.dofPerNode]

        self.residuum = 0
        for dof in range(0, len(displacements)):
            self.residuum += displacements[dof]**2
            
            
    def updateNodes(self):
        for node in self.nodes:
            node.updatePosition()


    def getResiduum(self):
        print(self.residuum)
        return self.residuum


    def printNodalCoordinates(self):
        for node in self.nodes:
            node.printCoordinates()


    def writeValues(self, filename):
        rounding = 10
        
        outputFile = open(filename, "w")
        outputFile.write("# vtk DataFile Version 2.0\n")
        outputFile.write("Results\n")
        outputFile.write("ASCII\n")
        outputFile.write("DATASET UNSTRUCTURED_GRID\n")
        outputFile.write("POINTS         " + str(self.numberOfNodes) + "           float\n")

        for node in self.nodes:
            for dof in range(0, 3):
                outputFile.write(str(node.referencePosition[dof]) + "\t")
            outputFile.write("\n")

        outputFile.write("\nCELLS\t" + str(self.numberOfElements) + "\t72\n")

        for element in self.elementGeometry:
            outputFile.write(" " + str(self.numberOfElements) + "\t")
            for entry in element:
                outputFile.write(str(entry) + "\t")
            outputFile.write("\n")
        outputFile.write("\n")

        outputFile.write("CELL_TYPES\t" + str(self.numberOfElements) + "\n")
        for i in range(0, self.numberOfElements):
            outputFile.write(" 12\n") 

        outputFile.write("\nPOINT_DATA\t" + str(self.numberOfNodes) + "\n")
        
        # Uncommented block
        for dof in range(0, self.dofPerNode):
            outputFile.write("SCALARS\tu" + str(dof+1) + "\tfloat 1\n")
            outputFile.write("LOOKUP_TABLE\tdefault\n")
            
            for node in self.nodes:
                outputFile.write(str(round(node.displacement[dof], 3)) + "\n")
            outputFile.write("\n")
        #for disp in range(0, 3):
        outputFile.write("VECTORS\tu \tfloat \n")
            
        for node in self.nodes:
            outputFile.write(str(round(node.displacement[0], rounding)) + "\t" \
                           + str(round(node.displacement[1], rounding)) + "\t" \
                           + str(round(node.displacement[2], rounding)) + "\n")
        outputFile.write("\n")
        
        outputFile.write("VECTORS\tv \tfloat \n")
            
        for node in self.nodes:
            outputFile.write(str(round(node.velocity[0], rounding)) + "\t" \
                           + str(round(node.velocity[1], rounding)) + "\t" \
                           + str(round(node.velocity[2], rounding)) + "\n")
        outputFile.write("\n")
        
        outputFile.write("VECTORS\ta \tfloat \n")
            
        for node in self.nodes:
            outputFile.write(str(round(node.acceleration[0], rounding)) + "\t" \
                           + str(round(node.acceleration[1], rounding)) + "\t" \
                           + str(round(node.acceleration[2], rounding)) + "\n")
        outputFile.write("\n")
        
        outputFile.write("SCALARS\ttheta\tfloat 1\n")
        outputFile.write("LOOKUP_TABLE\tdefault\n")
            
        for node in self.nodes:
            outputFile.write(str(round(node.displacement[3], rounding)) + "\n")
        outputFile.write("\n")

        for stre in range(0, 3):
            outputFile.write("SCALARS\tsig_" + str(stre+1)*2 + "\tfloat\t1\n")
            outputFile.write("LOOKUP_TABLE\tdefault\n")
            for node in self.nodes:
                outputFile.write(str(round(node.sigma[stre] / node.weightFactor, rounding)) + "\n")
            outputFile.write("\n")

        outputFile.write("SCALARS vonMises\tfloat\t1\n")
        outputFile.write("LOOKUP_TABLE\tdefault\n")

        for node in self.nodes:
            outputFile.write(str(round(node.vonMises / node.weightFactor, rounding)) + "\n")

        outputFile.close()
