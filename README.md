# FEM-Non-Linear

`fem-non-linear` is a Python-based implementation of the Finite Element Method (FEM) for solving non-linear problems in solid mechanics. This project supports material and geometric non-linearities, making it suitable for complex simulations involving large deformations and advanced material models.

## Features

- **Non-linear Material Models**: Supports hyperelasticity, thermal expansion, and other advanced material behaviors.
- **Geometric Non-linearities**: Handles large deformations and rotations.
- **Dynamic Analysis**: Implements the Newmark-beta method for time integration.
- **Customizable Mesh**: Supports cuboidal meshes with user-defined dimensions and element configurations.
- **Boundary Conditions**: Flexible application of Dirichlet and Neumann boundary conditions.
- **Post-Processing**: Outputs results in VTK format for visualization in tools like ParaView.

---

## Governing Equations

The project solves the coupled thermo-mechanical problem using the following governing equations:

1. **Momentum Balance**:
   \[
   \nabla \cdot \sigma + \mathbf{f} = \rho \ddot{\mathbf{u}}
   \]
   where:
   - \(\sigma\): Cauchy stress tensor
   - \(\mathbf{f}\): Body forces
   - \(\rho\): Density
   - \(\ddot{\mathbf{u}}\): Acceleration

2. **Energy Balance**:
   \[
   \rho c_p \dot{\theta} - \nabla \cdot (\kappa \nabla \theta) = Q
   \]
   where:
   - \(c_p\): Heat capacity
   - \(\theta\): Temperature
   - \(\kappa\): Thermal conductivity
   - \(Q\): Heat source

3. **Constitutive Model**:
   The stress-strain relationship is defined using a hyperelastic material model:
   \[
   \sigma = \lambda \text{tr}(\epsilon)I + 2\mu \epsilon
   \]
   where:
   - \(\lambda, \mu\): Lamé parameters
   - \(\epsilon\): Green-Lagrange strain tensor

---

## Project Structure

```
fem-non-linear/
│
├── main.py                # Main routine to run the simulation
├── inputs.py              # Input parameters for the simulation
├── mesh.py                # Mesh generation and handling
├── element.py             # Element-level computations
├── node.py                # Node-level computations
├── project_description.pdf # Detailed project description
├── Outputs/               # Directory for simulation results
└── README.md              # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib
- ParaView (for visualization)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fem-non-linear.git
   cd fem-non-linear
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Simulation

To run the simulation, execute the `main.py` file:
```bash
python main.py
```

### Input Parameters

The simulation parameters are defined in `inputs.py`. Key parameters include:

- **Material Properties**:
  ```python
  nu = 0.3  # Poisson's ratio
  emod = 210000.0  # Young's modulus
  ```

- **Mesh Configuration**:
  ```python
  length = 100.0  # Length of the cuboid
  width = 80.0    # Width of the cuboid
  height = 140.0  # Height of the cuboid
  ```

- **Boundary Conditions**:
  ```python
  fixedDofs = {
      0: [0, 1, 2],  # Node 0 fixed in x, y, z
      1: [1, 2],     # Node 1 fixed in y, z
  }
  ```

### Output

Simulation results are saved in the `Outputs/` directory in VTK format. These files can be visualized using ParaView.

---

## Code Overview

### 1. `main.py`

The main routine initializes the mesh, applies boundary conditions, and iteratively solves the non-linear system using the Newton-Raphson method.

```python
geometry = Cuboid(
    numberOfElements, numberOfNodes, nodesPerElement, dofPerNode,
    numberOfGaussPoints, fixedDofs, length, width, height, initValues,
    theta0, alphaNM, betaNM, timeStep, lmu, llambda, alphaS, kappa,
    alphaT, cp, rho, rT
)

for time in range(0, simulationTime, timeStep):
    for newtonIteration in range(0, maxNewtonIterations):
        geometry.resetGlobalStiffness()
        geometry.resetGlobalForces()
        geometry.computeShapeFunctions()
        geometry.computeStresses()
        geometry.applyBoundaryConditions()
        geometry.computeDisplacements()
```

### 2. `mesh.py`

Defines the `Cuboid` class for generating a structured mesh of nodes and elements.

```python
class Cuboid:
    def __init__(self, numberOfElements, numberOfNodes, ...):
        self.nodes = []
        self.elements = []
        for z in linspace(0, height, nodesPerDim):
            for y in linspace(0, width, nodesPerDim):
                for x in linspace(0, length, nodesPerDim):
                    self.nodes.append(Node(dofPerNode, [x, y, z, theta0], ...))
```

### 3. `element.py`

Handles element-level computations, including stiffness matrix assembly and stress evaluation.

```python
def computeStiffness(self):
    for gp in range(0, self.numberOfGaussPoints):
        for node1 in range(0, self.numberOfNodes):
            for node2 in range(0, self.numberOfNodes):
                self.stiffness[node1][node2] += ...
```

### 4. `node.py`

Defines the `Node` class for storing nodal information such as displacements, velocities, and forces.

```python
class Node:
    def __init__(self, dof, position, initValues, ...):
        self.displacement = array(initValues)
        self.velocity = zeros(dof)
        self.acceleration = zeros(dof)
```

---

## Example

### Problem Setup

Simulate a cuboid under thermal and mechanical loads with the following parameters:

- **Material Properties**:
  - Young's modulus: \(E = 210 \, \text{GPa}\)
  - Poisson's ratio: \(\nu = 0.3\)

- **Boundary Conditions**:
  - Fixed nodes on one face.
  - Uniform load applied on the opposite face.

### Running the Simulation

Modify `inputs.py` to set up the problem, then run:
```bash
python main.py
```

### Visualizing Results

Open the output `.vtk` files in ParaView to visualize displacements, stresses, and temperature distributions.

---

## Future Work

- Extend support for additional element types (e.g., tetrahedral, hexahedral).
- Implement additional material models (e.g., plasticity, viscoelasticity).
- Add parallelization for large-scale simulations.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Special thanks to the open-source community for providing tools and libraries that made this project possible.