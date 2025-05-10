### Parameter declarations

# numerical parameters
alphaNM = 1.0 / 4.0 
betaNM =  1.0 / 2.0
residualPrecision = 1E-6

# material parameters
nu = 0.3                                        # Poisson's ratio
emod = 210000.0                                 # Young's modulus
llambda = emod*nu/((1.0-2.0*nu)*(1.0+nu))       # 1st Lamé constant
lmu = emod/(2.0 + 2.0*nu)                       # 2nd Lamé constant
alphaS = 0.0001                                 # thermal expansion coefficient
kappa = 2.0 / 3.0 * lmu + llambda               # compression modulus
alphaT = 35.0                                   # thermal conductivity
cp = 480000000.0                                # heat capacity, constant pressure
rho = 0.0000000075                              # initial density
rT = 0.0                                        # external heat source

# problem instance
length = 100.0                                  # dimensions of the cuboid 
width = 80.0          
height = 140.0
theta0 = 300.0                                  # initial temperature
initValues = [0.0, 0.0, 0.0, theta0]            # initial displacements and temperature
numberOfElements = 8
numberOfNodes = 27
dofPerNode = 4
nodesPerElement = 8
numberOfGaussPoints = 8
simulationTime = 300#500      # load steps
timeStep = 1              # step size
maxNewtonIterations = 100#10
#load = -1000000.0


### boundary conditions

# node : [list, of, dofs]
fixedDofs = {\
#        0 : [2], \
#        1 : [0, 2], \
#        2 : [2], \
#        3 : [1, 2], \
#        4 : [0, 1, 2], \
#        5 : [2], \
#        6 : [2], \
#        7 : [2], \
#        8 : [2]}

        0 : [0, 1, 2], \
        1 : [1, 2], \
        2 : [1, 2], \
        3 : [0, 2], \
        4 : [2], \
        5 : [2], \
        6 : [0, 2], \
        7 : [2], \
        8 : [2]}

# forces
# node: [list, of, forces per dof]
heatFlux = 0.0 #-500000.0
fz = 100000.0
externalForces = {\
        18 : [0, 0, fz/4/4, heatFlux / 4 / 4], \
        19 : [0, 0, fz/4/2, heatFlux / 4 / 2], \
        20 : [0, 0, fz/4/4, heatFlux / 4 / 4], \
        21 : [0, 0, fz/4/2, heatFlux / 4 / 2], \
        22 : [0, 0, fz/4/1, heatFlux / 4 / 1], \
        23 : [0, 0, fz/4/2, heatFlux / 4 / 2], \
        24 : [0, 0, fz/4/4, heatFlux / 4 / 4], \
        25 : [0, 0, fz/4/2, heatFlux / 4 / 2], \
        26 : [0, 0, fz/4/4, heatFlux / 4 / 4]}  

#externalForces = {\
#        18 : [0, 0, fz/4/4, heatFlux / 4 / 4], \
#        19 : [0, 0, fz/4/2, heatFlux / 4 / 2], \
#        20 : [0, 0, fz/4/4, heatFlux / 4 / 4], \
#        21 : [0, 0, 0, heatFlux / 4 / 2], \
#        22 : [0, 0, 0, heatFlux / 4 / 1], \
#        23 : [0, 0, 0, heatFlux / 4 / 2], \
#        24 : [0, 0, 0, heatFlux / 4 / 4], \
#        25 : [0, 0, 0, heatFlux / 4 / 2], \
#        26 : [0, 0, 0, heatFlux / 4 / 4]}  