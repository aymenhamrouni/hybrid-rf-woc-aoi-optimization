# Hybrid RF-WOC Network Optimization with Age of Information (AoI) Metrics

This project implements a Mixed Integer Non-Linear Programming (MINLP) model for optimizing network scheduling in hybrid Radio Frequency (RF) and Wireless Optical Communication (WOC) systems, with a focus on Age of Information (AoI) metrics. The implementation is based on the research presented in the following paper:

> Hamrouni, A., Pollin, S., & Sallouha, H. (2024). AoI in Context-Aware Hybrid Radio-Optical IoT Networks. In *2024 IEEE Global Communications Conference (GLOBECOM)* (pp. 1966-1972). IEEE. DOI: [10.1109/GLOBECOM52923.2024.10901639](https://doi.org/10.1109/GLOBECOM52923.2024.10901639)

## Project Overview

This implementation provides a comprehensive framework for optimizing hybrid RF-WOC networks with multiple objectives and constraints. The framework includes:

### Core Features

1. **Multiple Optimization Objectives**:
   - `max_messages`: Maximize the number of transmitted messages
   - `min_energy`: Minimize energy consumption
   - `min_delay`: Minimize transmission delay
   - `min_switching`: Minimize technology switching overhead
   - `max_messages-w.r.t.energy`: Maximize messages with respect to energy consumption
   - `multi-objective-max-messages and min-energy`: Combined objective with weighted factors

2. **Network Analysis Capabilities**:
   - Energy efficiency evaluation
   - Age of Information (AoI) metrics:
     - Peak AoI
     - Mean AoI
     - AoI for communicating pairs
     - AoI for all network pairs
   - Packet delivery performance
   - Network throughput analysis

3. **Simulation Types**:
   - Variable network load size w.r.t energy consumption and packet delivery rate
   - Variable network size analysis:
     - Varying number of Access Points (APs)
     - Varying number of IoT devices

4. **Visualization Tools**:
   - Packet delivery rate and throughput
   - Energy consumption patterns
   - AoI evolution over time
   - Technology switching patterns

### Advanced Features

1. **Constraint Handling**:
   - Network topology constraints
   - Energy constraints
   - Signal quality requirements (SNR thresholds)
   - Message scheduling constraints
   - Technology switching constraints

2. **Performance Metrics**:
   - Energy per bit consumption
   - Total exchanged packets
   - Peak and Mean AoI
   - Network throughput
   - Communication delay
   - Technology switching frequency

## Project Structure

```
.
├── main.py                 # Main simulation driver and entry point
│   - Handles simulation setup and execution
│   - Manages Monte Carlo simulations
│   - Controls network size variations
│   - Coordinates different simulation types
│
├── MINLP_model.py         # MINLP optimization model implementation
│   - Implements the core optimization model
│   - Handles multiple objective functions
│   - Manages constraints and variables
│   - Solves the optimization problem
│
├── data_generation.py     # Network and message data generation
│   - Generates network topology
│   - Creates message queues and priorities
│   - Calculates channel parameters
│   - Manages network characteristics
│
├── utils.py               # Utility functions and metrics
│   - Provides visualization tools
│   - Implements AoI calculations
│   - Handles simulation results processing
│   - Contains helper functions for analysis
│
├── constants.py           # System constants and parameters
│   - Physical layer constants
│   - WOC and RF parameters
│   - System configuration values
│   - Simulation parameters
│
├── requirements.txt       # Python dependencies
│   - Lists all required packages
│   - Specifies version requirements
│
└── .gitignore            # Git ignore rules
    - Excludes Python cache files
    - Ignores virtual environment
    - Excludes IDE-specific files
```

## Dependencies

- Python 3.8+
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- Docplex >= 2.25.236 (Academic CPLEX for large-scale optimization)
- Pandas >= 1.3.0
- Seaborn >= 0.11.0
- SciPy >= 1.7.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aymenhamrouni/hybrid-rf-woc-aoi-optimization.git
cd hybrid-rf-woc-aoi-optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation

1. Configure system parameters in `constants.py`:
```python
# System parameters
Time_unit = 5e-5  # 1 unit is 50 microseconds
Time = 10         # Simulation duration
N_d = 5          # Number of IoT nodes
N_APs = 2        # Number of Access Points
S_p = 128        # Packet size in bits
snr_min_woc = 10 # Minimum SNR for WOC (dB)
snr_min_rf = 30  # Minimum SNR for RF (dB)
```

2. Configure simulation output parameters:
```python
# Output control parameters
Verbose = True   # Enable detailed simulation output
VIZ = True       # Enable visualization plots
```

3. Run the main simulation:
```bash
python main.py
```

### Output Control Parameters

The simulation provides two key parameters to control the level of detail in the output:

1. **Verbose Mode** (`Verbose`):
   - When `True`: Provides detailed simulation output including:
     - Optimization progress
     - Network statistics
     - Energy consumption details
     - AoI metrics for each node pair
     - Technology switching information
     - Performance metrics
   - When `False`: Provides minimal output, showing only essential information

2. **Visualization Mode** (`VIZ`):
   - When `True`: Generates various plots including:
     - Network parameter distributions (SNR and capacity)
     - Energy efficiency metrics
     - AoI evolution plots
     - Scheduling matrices
     - Technology switching patterns
     - Performance comparison plots
   - When `False`: Disables all visualization outputs

### Advanced Usage

1. **Network Size Analysis**:
```python
from utils import run_APs_simulation, run_Nd_simulation

# Varying number of APs
run_APs_simulation(
    N_APs_range=range(2, 10),
    N_d=5,
    fixed_size=200*128,  # Packet size
    minD=5, maxD=20,     # Distance range
    minE=0.05, maxE=0.1, # Energy range
    Time=10,             # Time slots
    N_montecarlo=2,      # Monte Carlo runs
    Verbose=True,        # Enable detailed output
    VIZ=True            # Enable visualizations
)

# Varying number of devices
run_Nd_simulation(
    N_d_range=range(5, 10),
    N_APs=3,
    # ... other parameters ...
    Verbose=True,        # Enable detailed output
    VIZ=True            # Enable visualizations
)
```

2. **Custom Optimization Objectives**:
```python
from MINLP_model import MINLP_model

# Run optimization with specific objective
solution = MINLP_model(
    # ... network parameters ...
    objective_type='multi-objective-max-messages and min-energy',
    Timeout=60,
    Verbose=True,        # Enable detailed output
    VIZ=True            # Enable visualizations
)
```

## Output and Visualization

The simulation generates several visualizations when `VIZ=True`:
1. Network parameter distributions (SNR and capacity)
2. Energy efficiency metrics
3. AoI evolution plots
4. Scheduling matrices
5. Performance comparison plots

When `Verbose=True`, the simulation provides detailed console output including:
1. Optimization progress and solver details
2. Network statistics and performance metrics
3. Energy consumption patterns
4. AoI metrics for each node pair
5. Technology switching information

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This code is provided for research purposes only. All rights reserved. Any use of this code must include proper citation of the original paper.

## Contact

For questions or inquiries, please contact:
- Aymen Hamrouni: [aymen.hamrouni@kuleuven.be]
- Sofie Pollin: [sofie.pollin@kuleuven.be]
- Hazem Sallouha: [hazem.sallouha@kuleuven.be]
