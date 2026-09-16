# UWMSN-SIM

This package provides the kinematic simulation environment for the UWMSN research project. It models AUV motion, target dynamics, communication-related state updates, and experiment logging needed to evaluate underwater tracking and monitoring strategies.

## Purpose

`uwmsn-sim` is the core simulation layer of the overall system. It generates team behavior, publishes the current target and vehicle states, and records trajectories and statistics used by the rest of the codebase.

## Main modules

- `src/main-kinematic_MTT.py`: main simulator entry point for the multi-target scenario.
- `src/auv_node_MTT.py`: AUV simulation node and team-state handling.
- `src/Classes/`: sensor, estimation, kinematic, and utility models.
- `launch/`: startup configurations for different team sizes and optimization modes.
- `plot_*.py`: plotting and result-analysis utilities.
- `relevant_logs/`: saved outputs and representative experimental data.

## Simulation workflow

The simulator publishes state information, receives the current control policy from the optimization layer, updates the AUV positions, and stores the resulting trajectories for later analysis. It is designed to work together with the acoustic communication layer and the motion-optimization stack.

## Requirements

- ROS 1 catkin workspace
- `rospy`
- `std_msgs`
- `numpy`
- `scipy`
- `matplotlib`
- custom `uwmsn_msgs` dependency if used in your setup

## Installation

Place the package inside the `src` folder of a ROS workspace and build the full workspace:

```bash
cd ~/ros1_ws
catkin_make
source devel/setup.bash
```

## Usage

Start the simulator with the standard launch file:

```bash
roslaunch uwmsn-sim simulation_node_startup.launch auvNum:=4 targetNum:=1 formationControl:=0
```

If you are using the multi-target setup, make sure the arguments match the corresponding launch file and the optimization/controller setup.

## Reproducibility

To reproduce simulation results reliably:

1. Keep the AUV count, target count, and formation arguments consistent.
2. Use the same launch file and control parameters for all runs in a series.
3. Save outputs in a dedicated experiment directory.
4. Regenerate plots using the scripts in this package after each run.

Example:

```bash
cd ~/ros1_ws/src/uwmsn-sim
python3 plot.py
python3 plot_results_MTT.py
python3 plot_connectivity.py
```

## Citation

If this package contributes to your research, please cite the project publication describing the motion optimization strategy for passive acoustic monitoring with a team of AUVs under intermittent communication.

> Tiranti, A., et al. "Motion optimization strategy for passive acoustic monitoring with a team of AUVs considering intermittent communication." Please cite the published paper appropriately in any derived work.

## Notes

- This package is intended for ROS 1 and a catkin workspace.
- It depends on the message definitions and runtime environment used by the rest of the project.
- It is suitable for research reproduction and comparative evaluation in an academic setting.
