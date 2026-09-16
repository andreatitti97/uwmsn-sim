# UWMSN-SIM

This package provides the kinematic simulation environment for the UWMSN research project. It models AUV motion, target dynamics, communication-related state updates, and experiment logging needed to evaluate underwater tracking and monitoring strategies.

## Purpose

`uwmsn-sim` is the core simulation layer of the overall system. It generates the team behavior, publishes the current target and vehicle states, and records trajectories and statistics used by the rest of the codebase.

## Main modules

- `src/main-kinematic_MTT.py`: main simulator entry point for the multi-target scenario.
- `src/auv_node_MTT.py`: AUV simulation node and team state handling.
- `src/Classes/`: sensor, estimation, kinematic, and utility models.
- `launch/`: startup configurations for different team sizes and optimization modes.
- `plot_*.py`: plotting and result-analysis utilities.
- `relevant_logs/`: saved outputs and representative experimental data.

## Simulation workflow

The simulator publishes state information, receives the current control policy from the optimization layer, updates the AUV positions, and stores the resulting trajectories for later analysis. It is designed to work together with the acoustic communication layer and the motion optimization stack.

## Citation

If this package contributes to your research, please cite the project publication describing the motion optimization strategy for passive acoustic monitoring with a team of AUVs under intermittent communication.

> Tiranti, A., et al. "Motion optimization strategy for passive acoustic monitoring with a team of AUVs considering intermittent communication." Please cite the published paper appropriately in any derived work.

A second paper placeholder may be added here once the final metadata is ready.

## Notes

- This package is intended for ROS 1 and a catkin workspace.
- It depends on the `uwmsn_msgs` message definitions and interacts with the other packages in this project.
- The repository is in a research-ready but still evolving state before the first public GitHub release.
