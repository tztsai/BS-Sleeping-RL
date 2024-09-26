# Energy-Efficient Collaborative Base Station Control in Massive MIMO Cellular Networks

This repository is associated with the publication "[Multi-agent Reinforcement Learning for Energy Saving in Multi-Cell Massive MIMO Systems](https://ieeexplore.ieee.org/document/10624787)". This work provides a Multi-Agent Reinforcement Learning (MARL) approach to minimize the total energy consumption of multiple massive MIMO base stations (BSs) in a multi-cell network, while maintaining overall quality-of-service.

The strategy involves making decisions on multi-level advanced sleep modes, antenna switching, and user association for the base stations. By modelling the problem as a decentralized partially observable Markov decision process (DEC-POMDP), we propose a multi-agent proximal policy optimization (MAPPO) algorithm to obtain a collaborative BS control policy.

## Overview

This solution has been shown to significantly improve network energy efficiency, adaptively switch the BSs into different depths of sleep, reduce inter-cell interference, and maintain a high quality-of-service (QoS). 

## Explanation of the Project Structure

`train.sh` is the bash script for training the MAPPO agent. The hyparameters of the agent can be set in this script. Set the values of `wandb_user` and `wandb_api_key` if you want to log your training using `wandb`. In addition, the config of the cellular network environment can be found at `network/config.py`.

After training, the model will be stored in the `results` folder. Then the `simulate.sh` script can be used to run simulation of the agent interacting with the cellular network. The results of the simulation will be stored in the `analysis/sim_stats` folder. You can also benchmark the performance of several trained models using `analysis/benchmark.py`. It will produce plots in several `*_plots/` folders, similar to those in the published paper.

## Features

The key features of the project include:

- Simulating a 5G network environment using real-world mobile traffic patterns.
- Implementing a multi-agent proximal policy optimization (MAPPO) algorithm for collaborative base station control.
- Ensuring that the algorithm results in significant energy savings compared to baseline solutions, without compromising on QoS.

## Environment Configuration

The configuration of the simulation environment is as follows:

### Traffic Model

The traffic generator uses real-world data and contains arrival rates for each time slot (of 20 mins) and each application category.

### Action Space

- **Switch Antennas**: Options include -4, 0, +4.
- **Switch Sleep Mode**: Options include active (0), SM1 (1ms activation delay) (1), SM2 (10ms activation delay) (2), and SM3 (100ms activation delay) (3).
- **Switch Connection Mode**: Options include disconnecting all users (0), keeping current connections but refusing new connections (1), and accepting new connections (2).

### State Space

The state of the environment is defined by:

- Total power consumption.
- User statistics.
- Actual and required sum rates.
- State of the base stations, which includes:
  - Power consumption.
  - Number of active antennas.
  - Connection mode.
  - Sleep mode.
  - Next sleep mode.
  - Remaining wake-up time.
  - History of traffic rates.
  - Associated user statistics.

### Reward

The reward function is a combination of
- Weighted sum of drop rates in each application category.
- Total ower consumption.

### Notes

- The agents make a decision every 20ms.
- When a base station is in SM1 and a new user arrives, it will wake up automatically.

## Contributions and Feedback

Feel free to provide feedback or contribute to this project. You can either fork this repository or open a new issue if you find a bug or have a feature request.
