# Hybrid RF-WOC Network Optimization with Age of Information (AoI) Metrics

This project implements a Mixed Integer Non-Linear Programming (MINLP) model for optimizing network scheduling in hybrid Radio Frequency (RF) and Wireless Optical Communication (WOC) systems, with a focus on Age of Information (AoI) metrics. The implementation is based on the research presented in the following paper:

> Hamrouni, A., Pollin, S., & Sallouha, H. (2024). AoI in Context-Aware Hybrid Radio-Optical IoT Networks. In *2024 IEEE Global Communications Conference (GLOBECOM)* (pp. 1966-1972). IEEE. DOI: [10.1109/GLOBECOM52923.2024.10901639](https://doi.org/10.1109/GLOBECOM52923.2024.10901639)

## Project Overview

This implementation provides a comprehensive framework for optimizing hybrid RF-WOC networks with multiple objectives and constraints. The framework includes:

### Core Features

1. **Multiple Optimization Objectives**:
   - `max_messages`: Maximize the number of transmitted messages
   - `min_energy`: Minimize energy consumption
   - `max_capacity`: Maximize network capacity
   - `min_delay`: Minimize transmission delay
   - `min_switching`: Minimize technology switching overhead
   - `max_messages-w.r.t.energy`: Maximize messages with respect to energy consumption
   - `multi-objective-max-messages and min-energy`: Combined objective with weighted factors

2. **Network Analysis Capabilities**:
   - SNR and capacity distribution analysis
   - Energy efficiency evaluation
   - Age of Information (AoI) metrics:
     - Peak AoI
     - Mean AoI
     - AoI for communicating pairs
     - AoI for all network pairs
   - Packet delivery performance
   - Network throughput analysis

3. **Simulation Types**:
   - Fixed network topology simulation
   - Variable network size analysis:
     - Varying number of Access Points (APs)
     - Varying number of IoT devices
   - Monte Carlo simulations for statistical analysis

4. **Visualization Tools**:
   - Network parameter distributions
   - Energy consumption patterns
   - AoI evolution over time
   - Scheduling matrix visualization
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

3. **Application Support**:
   - Multi-application scenarios
   - Different message sizes and priorities
   - Context-aware communication

## Project Structure

```
.
├── main.py                 # Main simulation driver
├── MINLP_model.py         # MINLP optimization model implementation
├── data_generation.py     # Network and message data generation
├── utils.py               # Utility functions and metrics
├── constants.py           # System constants and parameters
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore rules
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

2. Run the main simulation:
```bash
python main.py
```

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
    N_montecarlo=2       # Monte Carlo runs
)

# Varying number of devices
run_Nd_simulation(
    N_d_range=range(5, 10),
    N_APs=3,
    # ... other parameters
)
```

2. **Custom Optimization Objectives**:
```python
from MINLP_model import MINLP_model

# Run optimization with specific objective
solution = MINLP_model(
    # ... network parameters ...
    objective_type='multi-objective-max-messages and min-energy',
    Timeout=60
)
```

## Output and Visualization

The simulation generates several visualizations:
1. Network parameter distributions (SNR and capacity)
2. Energy efficiency metrics
3. AoI evolution plots
4. Scheduling matrices
5. Performance comparison plots

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