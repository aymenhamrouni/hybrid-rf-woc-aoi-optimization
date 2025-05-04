# Hybrid RF-WOC Network Optimization with Age of Information (AoI) Metrics

This project implements a Mixed Integer Non-Linear Programming (MINLP) model for optimizing network scheduling in hybrid Radio Frequency (RF) and Wireless Optical Communication (WOC) systems, with a focus on Age of Information (AoI) metrics. The implementation is based on the research presented in the following paper:

> Hamrouni, A., Pollin, S., & Sallouha, H. (2024). AoI in Context-Aware Hybrid Radio-Optical IoT Networks. In *2024 IEEE Global Communications Conference (GLOBECOM)* (pp. 1966-1972). IEEE. DOI: [10.1109/GLOBECOM52923.2024.10901639](https://doi.org/10.1109/GLOBECOM52923.2024.10901639)

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@INPROCEEDINGS{10901639,
  author={Hamrouni, Aymen and Pollin, Sofie and Sallouha, Hazem},
  booktitle={GLOBECOM 2024 - 2024 IEEE Global Communications Conference}, 
  title={AoI in Context-Aware Hybrid Radio-Optical IoT Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1966-1972},
  keywords={Radio frequency;Schedules;Simulation;Smart homes;Pareto optimization;Throughput;Communications technology;Internet of Things;Wearable devices;Surges;IoT;Hybrid RF-OC;AoI;Optimization},
  doi={10.1109/GLOBECOM52923.2024.10901639}
}
```

## Project Overview

This implementation provides a comprehensive framework for:
- Hybrid RF-WOC network optimization
- Age of Information (AoI) analysis
- Energy efficiency evaluation
- Network performance simulation
- Monte Carlo analysis of network parameters

## Project Structure

```
.
└── conference/
    ├── main.py                 # Main simulation driver
    ├── MINLP_model.py          # MINLP optimization model implementation
    ├── data_generation.py      # Network and message data generation
    ├── utils.py                # Utility functions and metrics
    └── constants.py            # System constants and parameters
```

## Key Features

1. **Network Optimization**
   - Hybrid RF-WOC communication scheduling
   - Energy efficiency optimization
   - Throughput maximization
   - Age of Information minimization

2. **Simulation Capabilities**
   - Variable network size analysis (APs and devices)
   - Monte Carlo simulations
   - Performance metrics visualization
   - Network parameter analysis

3. **Performance Metrics**
   - Energy per bit consumption
   - Total packets exchanged
   - Peak and Mean Age of Information
   - Network throughput
   - Communication delay

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Docplex
- Pandas
- Seaborn

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
   - Network size (N_APs, N_d)
   - Time units and simulation duration
   - Packet sizes and transmission parameters
   - SNR thresholds for RF and WOC

2. Run the main simulation:
```bash
python conference/main.py
```

### Network Size Analysis

The code supports two types of network size analysis:

1. **Varying Number of APs**
```python
from utils import run_APs_simulation

run_APs_simulation(
    N_APs_range=range(1, 6),  # Range of APs to test
    N_d=5,                    # Fixed number of devices
    fixed_size=128,           # Fixed packet size
    minD=5, maxD=20,         # Distance range
    minE=0.05, maxE=0.1,     # Energy range
    Time=10,                  # Time slots
    N_montecarlo=2,          # Number of Monte Carlo runs
    Verbose=True,            # Print detailed information
    VIZ=True                 # Show plots
)
```

2. **Varying Number of Devices**
```python
from utils import run_Nd_simulation

run_Nd_simulation(
    N_d_range=range(5, 26, 5),  # Range of devices to test
    N_APs=5,                    # Fixed number of APs
    fixed_size=128,             # Fixed packet size
    minD=5, maxD=20,           # Distance range
    minE=0.05, maxE=0.1,       # Energy range
    Time=10,                    # Time slots
    N_montecarlo=2,            # Number of Monte Carlo runs
    Verbose=True,              # Print detailed information
    VIZ=True                   # Show plots
)
```

## Output

The simulation generates several visualizations:
1. Network parameter distributions (SNR and capacity)
2. Energy per bit vs. total exchanged packets
3. Mean and Peak Age of Information metrics
4. Network scheduling matrices

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

# Network Optimization with RF and WOC Technologies

This project implements a Mixed Integer Non-Linear Programming (MINLP) model for optimizing network scheduling in a hybrid RF-WOC (Radio Frequency - Wireless Optical Communication) system.

## Project Structure

```
conference/
├── MINLP_withNetwork.py    # Main simulation and optimization code
├── MINLP_model.py          # MINLP optimization model implementation
├── data_generation.py      # Network and message data generation
├── utils.py                # Utility functions and metrics
├── constants.py            # System constants and parameters
└── requirements.txt        # Project dependencies
```

## Key Components

1. **MINLP_withNetwork.py**
   - Main simulation driver
   - Handles network setup and optimization
   - Manages RF and WOC communication modes
   - Collects and analyzes performance metrics

2. **MINLP_model.py**
   - Implements the core MINLP optimization model
   - Handles constraints and objective functions
   - Manages communication scheduling

3. **data_generation.py**
   - Generates network characteristics
   - Creates message queues and timing information
   - Handles packet size generation

4. **utils.py**
   - Provides utility functions for metrics tracking
   - Handles result visualization
   - Manages simulation statistics

5. **constants.py**
   - Contains all system parameters
   - Physical layer parameters
   - Communication parameters
   - Simulation settings

## Key Features

- Hybrid RF-WOC communication optimization
- Energy efficiency analysis
- Age of Information (AoI) metrics
- Network performance evaluation
- Monte Carlo simulations

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Docplex
- Pandas
- Seaborn

## Usage

1. Configure system parameters in `constants.py`
2. Run the main simulation:
   ```bash
   python MINLP_withNetwork.py
   ```

## Output

The simulation generates:
- Energy consumption metrics
- Communication delay statistics
- Age of Information (AoI) measurements
- Network performance visualizations

## Author

Aymen Hamrouni 

# Hybrid RF-WOC Network Optimization with Age of Information (AoI) Metrics

This project implements a Mixed Integer Non-Linear Programming (MINLP) model for optimizing network scheduling in hybrid Radio Frequency (RF) and Wireless Optical Communication (WOC) systems, with a focus on Age of Information (AoI) metrics. The implementation is based on the research presented in the following paper:

> Hamrouni, A., Pollin, S., & Sallouha, H. (2024). AoI in Context-Aware Hybrid Radio-Optical IoT Networks. In *2024 IEEE Global Communications Conference (GLOBECOM)* (pp. 1966-1972). IEEE. DOI: [10.1109/GLOBECOM52923.2024.10901639](https://doi.org/10.1109/GLOBECOM52923.2024.10901639)

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@INPROCEEDINGS{10901639,
  author={Hamrouni, Aymen and Pollin, Sofie and Sallouha, Hazem},
  booktitle={GLOBECOM 2024 - 2024 IEEE Global Communications Conference}, 
  title={AoI in Context-Aware Hybrid Radio-Optical IoT Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1966-1972},
  keywords={Radio frequency;Schedules;Simulation;Smart homes;Pareto optimization;Throughput;Communications technology;Internet of Things;Wearable devices;Surges;IoT;Hybrid RF-OC;AoI;Optimization},
  doi={10.1109/GLOBECOM52923.2024.10901639}
}
```

## Project Overview

This implementation provides a comprehensive framework for:
- Hybrid RF-WOC network optimization
- Age of Information (AoI) analysis
- Energy efficiency evaluation
- Network performance simulation
- Monte Carlo analysis of network parameters

## Project Structure

```
.
├── main.py                 # Main simulation driver
├── MINLP_model.py          # MINLP optimization model implementation
├── data_generation.py      # Network and message data generation
├── utils.py                # Utility functions and metrics
└── constants.py            # System constants and parameters
```

## Key Features

1. **Network Optimization**
   - Hybrid RF-WOC communication scheduling
   - Energy efficiency optimization
   - Throughput maximization
   - Age of Information minimization

2. **Simulation Capabilities**
   - Variable network size analysis (APs and devices)
   - Monte Carlo simulations
   - Performance metrics visualization
   - Network parameter analysis

3. **Performance Metrics**
   - Energy per bit consumption
   - Total packets exchanged
   - Peak and Mean Age of Information
   - Network throughput
   - Communication delay

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Docplex
- Pandas
- Seaborn

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
   - Network size (N_APs, N_d)
   - Time units and simulation duration
   - Packet sizes and transmission parameters
   - SNR thresholds for RF and WOC

2. Run the main simulation:
```bash
python main.py
```

### Network Size Analysis

The code supports two types of network size analysis:

1. **Varying Number of APs**
```python
from utils import run_APs_simulation

run_APs_simulation(
    N_APs_range=range(1, 6),  # Range of APs to test
    N_d=5,                    # Fixed number of devices
    fixed_size=128,           # Fixed packet size
    minD=5, maxD=20,         # Distance range
    minE=0.05, maxE=0.1,     # Energy range
    Time=10,                  # Time slots
    N_montecarlo=2,          # Number of Monte Carlo runs
    Verbose=True,            # Print detailed information
    VIZ=True                 # Show plots
)
```

2. **Varying Number of Devices**
```python
from utils import run_Nd_simulation

run_Nd_simulation(
    N_d_range=range(5, 26, 5),  # Range of devices to test
    N_APs=5,                    # Fixed number of APs
    fixed_size=128,             # Fixed packet size
    minD=5, maxD=20,           # Distance range
    minE=0.05, maxE=0.1,       # Energy range
    Time=10,                    # Time slots
    N_montecarlo=2,            # Number of Monte Carlo runs
    Verbose=True,              # Print detailed information
    VIZ=True                   # Show plots
)
```

## Output

The simulation generates several visualizations:
1. Network parameter distributions (SNR and capacity)
2. Energy per bit vs. total exchanged packets
3. Mean and Peak Age of Information metrics
4. Network scheduling matrices

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

# Network Optimization with RF and WOC Technologies

This project implements a Mixed Integer Non-Linear Programming (MINLP) model for optimizing network scheduling in a hybrid RF-WOC (Radio Frequency - Wireless Optical Communication) system.

## Project Structure

```
conference/
├── MINLP_withNetwork.py    # Main simulation and optimization code
├── MINLP_model.py          # MINLP optimization model implementation
├── data_generation.py      # Network and message data generation
├── utils.py                # Utility functions and metrics
├── constants.py            # System constants and parameters
└── requirements.txt        # Project dependencies
```

## Key Components

1. **MINLP_withNetwork.py**
   - Main simulation driver
   - Handles network setup and optimization
   - Manages RF and WOC communication modes
   - Collects and analyzes performance metrics

2. **MINLP_model.py**
   - Implements the core MINLP optimization model
   - Handles constraints and objective functions
   - Manages communication scheduling

3. **data_generation.py**
   - Generates network characteristics
   - Creates message queues and timing information
   - Handles packet size generation

4. **utils.py**
   - Provides utility functions for metrics tracking
   - Handles result visualization
   - Manages simulation statistics

5. **constants.py**
   - Contains all system parameters
   - Physical layer parameters
   - Communication parameters
   - Simulation settings

## Key Features

- Hybrid RF-WOC communication optimization
- Energy efficiency analysis
- Age of Information (AoI) metrics
- Network performance evaluation
- Monte Carlo simulations

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Docplex
- Pandas
- Seaborn

## Usage

1. Configure system parameters in `constants.py`
2. Run the main simulation:
   ```bash
   python MINLP_withNetwork.py
   ```

## Output

The simulation generates:
- Energy consumption metrics
- Communication delay statistics
- Age of Information (AoI) measurements
- Network performance visualizations

## Author

Aymen Hamrouni 