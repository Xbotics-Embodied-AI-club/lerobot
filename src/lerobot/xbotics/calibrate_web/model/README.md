# SO101 Robot - URDF and MuJoCo Description

This repository contains the URDF and MuJoCo (MJCF) files for the SO101 robot.

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

The MuJoCo file `scene.xml` supports two differenly calibrated SO101 robot files:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.xml`. 
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use -> `so101_old_calib.xml`.

To switch between calibration methods, modify the included robot file in `scene.xml`.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files. 

## Leader Model

`so101_leader_new_calib.urdf` and the matching `Wrist_Roll_SO101.stl`, `Handle_SO101.stl`, and `Trigger_SO101.stl` come from the official [SO-ARM100 Simulation/SO101 directory](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101). The leader uses a fixed handle and an independent trigger link instead of the follower jaw.

The upstream model marks the trigger zero, travel limits, and printed-part inertias as provisional estimates rather than measurements. Its 0–45 degree trigger range must be verified on the physical leader.

---

Feel free to open an issue or contribute improvements!

## Xbotics

网页标定的从臂改用星禾套件的模型（见 `kit/`），这里只留官方主臂 `so101_leader_new_calib.urdf` 与它用到的网格。
