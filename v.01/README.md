# Documentation for RF-OC MINLP Optimization for AoI

## Table of Contents
1. [Introduction](#introduction)
2. [Code Structure](#code-structure)
3. [Key Components](#key-components)
4. [Usage Guide](#usage-guide)
5. [Configuration](#configuration)
6. [Output Analysis](#output-analysis)
7. [Troubleshooting](#troubleshooting)

## Introduction

This documentation provides a detailed guide to the RF-OC MINLP Optimization framework for IoT networks. The framework implements a Mixed Integer Non-Linear Programming approach to optimize hybrid Radio Frequency (RF) and Optical Communication (OC) systems, focusing on Age of Information (AoI) metrics.
This code is deprecated. Please refer to newer code availabe at https://github.com/aymenhamrouni/hybrid-rf-woc-aoi-optimization.
## Code Structure

The project consists of several key Python files:

### Main Files
- `main.py`: Entry point for the simulation
- `milpModel.py`: Implementation of the Mixed Integer Linear Programming model
- `paretoMILP.py`: Pareto optimization implementation
- `Pareto.py`: Pareto front analysis
- `data_generation.py`: Simulation data generation
- `utils.py`: Helper functions and utilities

## Key Components

### 1. Main Simulation (`main.py`)
The main simulation file handles:
- Simulation setup and initialization
- Monte Carlo simulations
- Network parameter configuration
- Result visualization and analysis

### 2. Optimization Model (`milpModel.py`)
The core optimization model includes:
- Objective function definitions
- Constraint implementations
- Variable declarations
- Solver configuration

### 3. Pareto Analysis (`Pareto.py`)
Handles:
- Trade-off analysis between RF and OC
- Multi-objective optimization
- Performance metric calculations

### 4. Data Generation (`data_generation.py`)
Responsible for:
- Network topology generation
- Channel parameter calculations
- Message queue management
- Network characteristic simulation

### 5. Utilities (`utils.py`)
Provides:
- Visualization tools
- AoI calculations
- Performance metrics
- Helper functions

## Usage Guide

### Basic Usage
1. Configure parameters in `constants.py`
2. Run the main simulation:
   ```bash
   python main.py
   ```
3. Generate Pareto analysis:
   ```bash
   python Pareto.py
   ```


## Output Analysis

### Simulation Results
1. Device Selection Patterns
2. AoI Performance Metrics
3. Energy Consumption Analysis
4. Pareto Front Analysis
5. Performance Plots

### Visualization
- Network parameter distributions
- Energy efficiency metrics
- AoI evolution plots
- Scheduling matrices
- Performance comparison plots

## Troubleshooting
### Getting Help
For additional support, contact:
- Aymen Hamrouni: [aymen.hamrouni@kuleuven.be]
- Sofie Pollin: [sofie.pollin@kuleuven.be]
- Hazem Sallouha: [hazem.sallouha@kuleuven.be]

## Citation

If you use this code in your research, please cite:
> Hamrouni, A., Pollin, S., & Sallouha, H. (2024). AoI in Context-Aware Hybrid Radio-Optical IoT Networks. In _2024 IEEE Global Communications Conference (GLOBECOM)_ (pp. 1966-1972). IEEE. DOI: 10.1109/GLOBECOM52923.2024.10901639 