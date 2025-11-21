from go2_robot_data import PinGo2Model
from reference_trajectory import ReferenceTraj
from locomotion_mpc import Locomotion_MPC
from gait import Gait
from plot_helper import plot_contact_forces, plot_traj_tracking, plot_mpc_result, plot_swing_foot_traj, plot_full_traj
import numpy as np
import matplotlib.pyplot as plt
from mujoco_model import MuJoCo_GO2_Model
import mujoco as mj
from leg_controller import LegController
import time
import mujoco.viewer as mjv

run_sim_length = 2 # Run 1s
go2 = PinGo2Model()
traj = ReferenceTraj()
gait = Gait(frequency_hz=3, duty=0.6)

# 1) MuJoCo Object
mujoco_go2 = MuJoCo_GO2_Model()
mujoco_go2.update_with_q_pin(go2.current_config.get_q())

mpc_i = 0
mpc_dt = 0.33/16 # 1/16 of a gait cycle
mpc_i_end = int(run_sim_length/mpc_dt)

leg_ctrl_i = 0
leg_ctrl_dt = 1/1000 # 1000 hz
leg_ctrl_i_end = int(run_sim_length/leg_ctrl_dt)

mpc_force_FL = np.zeros(3)
mpc_force_FR = np.zeros(3)
mpc_force_RL = np.zeros(3)
mpc_force_RR = np.zeros(3)

mpc_force_FL_long = np.zeros((3, leg_ctrl_i_end))
mpc_force_FR_long = np.zeros((3, leg_ctrl_i_end))
mpc_force_RL_long = np.zeros((3, leg_ctrl_i_end))
mpc_force_RR_long = np.zeros((3, leg_ctrl_i_end))

tau_FL = np.zeros((3, leg_ctrl_i_end))
tau_FR = np.zeros((3, leg_ctrl_i_end))
tau_RL = np.zeros((3, leg_ctrl_i_end))
tau_RR = np.zeros((3, leg_ctrl_i_end))

foot_pos_now = np.zeros((3, leg_ctrl_i_end))
foot_pos_des = np.zeros((3, leg_ctrl_i_end))
foot_vel_now = np.zeros((3, leg_ctrl_i_end))
foot_vel_des = np.zeros((3, leg_ctrl_i_end))

FL_foot_pos = np.zeros((3, leg_ctrl_i_end))

x_vec = np.zeros((12, leg_ctrl_i_end))

solve_time = []
X_opt = []
U_opt = []


# 1) Control Reference
x_vel_des = 0.0001
y_vel_des = 0.3
z_pos_des = 0.27
yaw_rate_des = 0

# 2) Simulation parameters
control_hz = 1/leg_ctrl_dt
mpc_hz = 1/mpc_dt
mujoco_go2.model.opt.timestep = 1.0 / control_hz
steps_per_mpc = max(1, int(control_hz // mpc_hz))

#
q = go2.current_config.get_q()
q[0]=-2
mujoco_go2.update_with_q_pin(q)

# 2) Compute the dynamics and desired trajectory
traj.generateConstantTraj(go2, gait, 0,
                        x_vel_des, y_vel_des, z_pos_des, yaw_rate_des, 
                        time_step=mpc_dt, time_horizon=0.5)
go2.update_dynamics(traj, mpc_dt)

# 3) Build an empty QP solver object
mpc = Locomotion_MPC(go2, traj)
leg_controller = LegController()

nq = mujoco_go2.model.nq

t_log = np.zeros(leg_ctrl_i_end)
q_log = np.zeros((leg_ctrl_i_end, nq))



# 4) Run simulation
print(f"Running simulation at {control_hz} Hz (dt={leg_ctrl_dt:.3f}s)...")
while leg_ctrl_i < leg_ctrl_i_end:

    time_now = mujoco_go2.data.time

    # 1) Update Pinocchio model with MuJuCo data
    mujoco_go2.update_pin_with_mujoco(go2)
    x_vec[:, leg_ctrl_i] = go2.compute_com_x_vec().reshape(-1)

    # --- minimal log for replay ---
    t_log[leg_ctrl_i]    = time_now
    q_log[leg_ctrl_i, :] = mujoco_go2.data.qpos
    # ------------------------------

    ## MPC LOOP
    if (leg_ctrl_i % steps_per_mpc) == 0:
        # 6) Update reference trajectory 
        traj.generateConstantTraj(go2, gait, time_now, 
                                x_vel_des, y_vel_des, z_pos_des, yaw_rate_des, 
                                time_step=mpc_dt, time_horizon=0.5)
        
        # 7) Update dynamics 
        go2.update_dynamics(traj, mpc_dt)
        
        # 8) Solve the QP with the latest states
        sol = mpc.solve_QP(go2, traj)
        solve_time.append(mpc.solve_time)

        # 9) Retrieve results
        N = traj.N
        w_opt = sol["x"].full().flatten()

        X_opt = w_opt[: 12*(N)].reshape((12, N), order='F')
        U_opt = w_opt[12*(N):].reshape((12, N), order='F')

        mpc_force_FL = U_opt[0:3, 0]
        mpc_force_FR = U_opt[3:6, 0]
        mpc_force_RL = U_opt[6:9, 0]
        mpc_force_RR = U_opt[9:12, 0]
        # Only apply the first input

    mpc_force_FL_long[:,leg_ctrl_i] = mpc_force_FL
    mpc_force_FR_long[:,leg_ctrl_i] = mpc_force_FR
    mpc_force_RL_long[:,leg_ctrl_i] = mpc_force_RL
    mpc_force_RR_long[:,leg_ctrl_i] = mpc_force_RR

    [tau_FL[:,leg_ctrl_i], foot_pos_des[:,leg_ctrl_i], foot_pos_now[:,leg_ctrl_i], foot_vel_des[:,leg_ctrl_i], foot_vel_now[:,leg_ctrl_i]] = leg_controller.compute_FL_torque(go2, gait, mpc_force_FL, time_now, leg_ctrl_dt)
    tau_FR[:,leg_ctrl_i] = leg_controller.compute_FR_torque(go2, gait, mpc_force_FR, time_now, leg_ctrl_dt)
    tau_RL[:,leg_ctrl_i] = leg_controller.compute_RL_torque(go2, gait, mpc_force_RL, time_now, leg_ctrl_dt)
    tau_RR[:,leg_ctrl_i] = leg_controller.compute_RR_torque(go2, gait, mpc_force_RR, time_now, leg_ctrl_dt)

    max = 45
    tau_FL[:, leg_ctrl_i] = np.clip(tau_FL[:, leg_ctrl_i], -max, max)
    tau_FR[:, leg_ctrl_i] = np.clip(tau_FR[:, leg_ctrl_i], -max, max)
    tau_RL[:, leg_ctrl_i] = np.clip(tau_RL[:, leg_ctrl_i], -max, max)
    tau_RR[:, leg_ctrl_i] = np.clip(tau_RR[:, leg_ctrl_i], -max, max)

    mj.mj_step1(mujoco_go2.model, mujoco_go2.data)
    mujoco_go2.set_leg_joint_torque("FL", tau_FL[:, leg_ctrl_i])
    mujoco_go2.set_leg_joint_torque("FR", tau_FR[:, leg_ctrl_i])
    mujoco_go2.set_leg_joint_torque("RL", tau_RL[:, leg_ctrl_i])
    mujoco_go2.set_leg_joint_torque("RR", tau_RR[:, leg_ctrl_i])
    mj.mj_step2(mujoco_go2.model, mujoco_go2.data)

    leg_ctrl_i += 1


print("Simulation ended")

t_vec = np.arange(leg_ctrl_i_end) * leg_ctrl_dt
force_all = np.vstack((mpc_force_FL_long, mpc_force_FR_long, mpc_force_RL_long, mpc_force_RR_long))
tau_all = np.vstack((tau_FL, tau_FR, tau_RL, tau_RR))
plot_mpc_result(t_vec, force_all, tau_all, x_vec, block=False)

model = mujoco_go2.model
data_replay = mj.MjData(model)

t_log = t_log[:leg_ctrl_i]   # just in case you didn't fill the whole buffer
q_log = q_log[:leg_ctrl_i]

render_hz = 120.0
render_dt_sim = 1.0 / render_hz
realtime_factor = 0.5

with mjv.launch_passive(model, data_replay) as viewer:
    while viewer.is_running():           # loop until the window is closed
        start_wall = time.perf_counter()
        t0 = t_log[0]
        next_render_t = t0

        k = 0
        T = len(t_log)

        # One full replay
        while k < T and viewer.is_running():
            t = t_log[k]

            # time to render a frame?
            if t >= next_render_t:
                data_replay.qpos[:] = q_log[k]
                mj.mj_forward(model, data_replay)
                viewer.sync()

                # real-time pacing
                target_wall = start_wall + (t - t0) / realtime_factor
                now = time.perf_counter()
                sleep_time = target_wall - now
                if sleep_time > 0:
                    time.sleep(sleep_time)

                next_render_t += render_dt_sim

            k += 1

        # when inner loop finishes, it just starts over from t_log[0]
        # outer while will keep looping until user closes the viewer window




# plot_swing_foot_traj(t_vec, foot_pos_now, foot_pos_des, foot_vel_now, foot_vel_des, False)

plt.plot(solve_time)
plt.axhline(y=33.33, color='r', linestyle='--', linewidth=1.5, label='30 Hz (33.33 ms)')
plt.axhline(y=20, color='g', linestyle='--', linewidth=1.5, label='50 Hz (20 ms)')
plt.xlabel("Iteration")
plt.ylabel("Time (ms)")
plt.title("QP Solve Time per Iteration")
plt.legend()
plt.grid(True)
plt.show(block=True)

# # print(traj.compute_x_ref_vec())

# # print(mujoco_go2.get_leg_joint_pos("FR"))
# # print(mujoco_go2.get_leg_joint_pos("FL"))
# print(traj.contact_table)

# print(np.shape(t_vec))



# # # # # 5) Run simulation with optimal input
[x_now, x_sim] = go2.run_simulation(U_opt)

pos_traj_sim = x_sim[0:3, :]
pos_traj_opt = X_opt[0:3, :]
x0_col = go2.compute_com_x_vec()
pos_traj_ref = np.hstack([x0_col[0:3, :], traj.compute_x_ref_vec()[0:3, :]])
traj_ref = np.hstack([x0_col, traj.compute_x_ref_vec()])

plot_full_traj(x_sim, traj_ref, block=True)
# plot_traj_tracking(pos_traj_ref, pos_traj_sim, block=True)
# # plot_traj_tracking(pos_traj_ref, pos_traj_opt, block=True)
# # # plot_contact_forces(U_opt, traj.contact_table, dt, block=True)

