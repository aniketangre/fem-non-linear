#-------------### This is the main routine
### Start with 'python main.py'

from inputs import *
from mesh import Cuboid 
#from time import sleep

geometry = Cuboid(numberOfElements, numberOfNodes, nodesPerElement, dofPerNode, numberOfGaussPoints, fixedDofs, \
            length, width, height, initValues, theta0, alphaNM, betaNM, timeStep, \
            lmu, llambda, alphaS, kappa, alphaT, cp, rho, rT)

def main():
#    geometry.printNodalCoordinates()

    # this starts the load step iteration
    for time in range(0, simulationTime, timeStep):    
        # in each step, the Newton-Raphson algorithm is performed until convergence
        for newtonIteration in range(0, maxNewtonIterations): 
            print(time, newtonIteration, maxNewtonIterations)
            
            # reset all global quantities
            geometry.resetGlobalStiffness()
            geometry.resetGlobalForces()
            geometry.resetStresses()      # includes von Mises
            geometry.resetWeightFactors()
            
            # Newmark algorithm, performed at each node
            geometry.computeVelocitiesAndAccelerations()  
            
            # resets all element stiffness matrices
            geometry.resetLocalStiffness()     
            # resets all element forces
            geometry.resetLocalForces()        
            
            # shape functions, isoJacobian/det/inv, Bmat
            geometry.computeShapeFunctions()   
            
            # computes gradU, gradUDot, theta, gradTheta, thetaDot etc.
            geometry.computeFieldVars()
            
            # computes stresses at Gauss points, then moves to nodes
            geometry.computeStresses()
    
            # compute the stiffness matrix for each element
            geometry.computeLocalStiffness()
            geometry.computeLocalForces()
            
            geometry.computeGlobalStiffness()
            #print(geometry.stiffness)  <---- debugging example
            
            # apply heat flux for first 100 load steps only
            if time * timeStep < 100:
                geometry.setExternalForces(externalForces)
            
            geometry.computeGlobalForces()
            
            # delete rows and columns of stiffness matrix and forces, apply external forces
            geometry.applyBoundaryConditions()
            #print(geometry.nodes[0].position)
            
            #print(geometry.elements[0].mC[0])
            # computes Displacements and saves to nodes
            geometry.computeDisplacements()
            
            # if precision better than set limit, stop Newton iteration
            if geometry.getResiduum() < residualPrecision:
                break 
        
        # update current coordinates and history variables
        geometry.updateNodes()
        #geometry.printNodalCoordinates()
                  
        geometry.writeValues("Outputs/outputfile" + str(time) + ".vtk")    


if __name__ == '__main__':
    main()
