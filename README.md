# Motion Control of Unitree Go2 Quadruped Robot

A **contact-force–optimization MPC locomotion controller** for the Unitree Go2 quadruped robot.  
Developed as part of the **UC Berkeley Master of Engineering (MEng)** capstone project in Mechanical Engineering.

---

## 🐾 Introduction

This repository contains a full implementation of a **Convex Model Predictive Controller (MPC)** for the Unitree Go2 quadruped robot.  
The controller is designed following the methodology described in the MIT publication:

> **"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"**  
> https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf

The objective of this project is to reproduce the main ideas presented in the paper—particularly the **contact-force MPC formulation**, convex optimization structure, and robust locomotion behavior—while integrating them into a modern, modular robotics control pipeline.

---

## 🔧 Libraries Used

- **MuJoCo** — fast, stable **physics simulation** used for testing locomotion, foot contacts, and dynamic behaviors.
- **Pinocchio** — efficient **kinematics and dynamics library** for:
  - forward kinematics  
  - Jacobians  
  - frame placements  
  - dynamics terms (M, C, g)

These libraries form the computational backbone of the control and simulation environment.

---

## 🦿 Controller Overview

Our motion control stack includes:

- **Centroidal MPC (50 Hz)**  
  Contact-force–based MPC implemented via **CasADi**, solving a convex quadratic program each control cycle.

- **Swing/Stance Leg Controller (1000 Hz)**  
  Performs foot trajectory tracking during swing and force tracking during stance.

- **Gait Scheduler**  
  Determines stance/swing timing and triggers trajectory generation.

- **Foot Trajectory Generator**  
  Computes smooth swing trajectories with configurable step height and duration.

---

## 🐍 Python Version Recommendation

- **Recommended Python version:** `3.10.12`  
  This project has been fully tested on Python 3.10.12—other 3.10 versions may work but are not guaranteed.

---
