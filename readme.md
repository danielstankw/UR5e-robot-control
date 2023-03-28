### General Information
This is code for operating Ur5e robot via Python script in 125[Hz].
The base of the code comes from RopeRobotics (https://github.com/martinbjerge/ur-interface)

### What can it be used for?
* Defines minimum jerk planner and designs minimum-jerk trajectory
* Implements PD and Impedance Controller for Peg-In-Hole tasks
* Includes 3 different modes of operation:
  * `run_robot.py`: allows for usage of impedance or PD control for insertion task
  * `run_robot_with_spiral.py`: allows for usage of spiral search or circular motion of the peg while in contact.
  * `run_robot_spiral_ml.py`: integrates trained model in Tensrflow for overlap detection -> incomplete

### Files explanation
1. `main.py`: main run file allowing for control of the robot. It defines 3 modes, described earlier as well as determined what type of controller will be used and what type of position error will be used.
2. `controller.py`: included impedance controller implementation and parameters
3. `helper_functions.py`: include files for data labeling, spiral search and circular motion variables.
4. `angle_transformation.py`: included various transformations used in robotic tasks.

#### Folders
1. `URBasic/urScript.py`: included various functions for robot communication. 
The functions need to follow UR format that can be found here: https://www.universal-robots.com/download/manuals-e-series/script/script-manual-e-series-sw-511/
2. `URBasic/urScriptExt.py`: included functions similarly to urScript.py however they are more advanced. This included `force_mode` and `servoj` which are
non-blocking functions allowing for real-time communication with the robot.
3. `Onrobot/Exconnector.py`: file used for utilizing OnRobot F/T sensor and reading the measurements.