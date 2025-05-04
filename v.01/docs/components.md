# Component Documentation

## 1. Main Simulation (`main.py`)

### Overview
The main simulation file orchestrates the entire optimization process, handling simulation setup, execution, and result analysis.

### Key Functions
- `run_MIP_simulation(OMEGA, n, t, nAPs)`: Main simulation function
- `plotWeightsFig(W1, W2, F, title)`: Visualization of weight analysis
- `plotAoI(d, name)`: Age of Information visualization

### Parameters
- `OMEGA`: Network configuration
- `n`: Number of devices
- `t`: Time slots
- `nAPs`: Number of Access Points

## 2. Optimization Model (`milpModel.py`)

### Overview
Implements the core Mixed Integer Linear Programming model for network optimization.

### Key Components
- Objective functions
- Constraints
- Variables
- Solver configuration

### Optimization Objectives
1. Maximize message transmission
2. Minimize energy consumption
3. Maximize network capacity
4. Minimize transmission delay
5. Minimize technology switching

## 3. Pareto Analysis (`Pareto.py`)

### Overview
Handles multi-objective optimization and trade-off analysis between RF and OC.

### Key Functions
- Pareto front generation
- Trade-off analysis
- Performance metric calculation

### Analysis Types
1. RF vs OC performance
2. Energy vs throughput
3. Delay vs capacity

## 4. Data Generation (`data_generation.py`)

### Overview
Generates simulation data including network topology and channel parameters.

### Key Functions
- Network topology generation
- Channel parameter calculation
- Message queue management
- Network characteristic simulation

### Parameters
- Network size
- Device distribution
- Channel conditions
- Message patterns

## 5. Utilities (`utils.py`)

### Overview
Provides helper functions and tools for analysis and visualization.

### Key Functions
- Visualization tools
- AoI calculations
- Performance metrics
- Helper functions

### Visualization Tools
1. Network parameter plots
2. Performance metrics
3. Optimization results
4. Trade-off analysis

## Common Parameters

### Network Parameters
- `N_d`: Number of IoT devices
- `N_APs`: Number of Access Points
- `S_p`: Packet size in bits
- `Time`: Simulation duration

### Channel Parameters
- `snr_min_woc`: Minimum SNR for WOC
- `snr_min_rf`: Minimum SNR for RF
- `channel_capacity`: Channel capacity calculations
- `interference_model`: Interference calculations

### Optimization Parameters
- `objective_type`: Optimization objective
- `constraints`: System constraints
- `variables`: Optimization variables
- `solver_parameters`: Solver configuration

## Performance Considerations

### Memory Usage
- Network size impact
- Monte Carlo iterations
- Data storage requirements

### Computation Time
- Solver configuration
- Network complexity
- Parallel processing options

### Visualization
- Plot size and complexity
- Data aggregation
- Memory management 