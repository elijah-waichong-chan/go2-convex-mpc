import numpy as np
from go2_robot_data import PinGo2Model
from gait import Gait
from dataclasses import dataclass

import time

class ReferenceTraj:

    def __init__(self):

        self.time: np.ndarray = np.empty(0)

        # These are from N = 1 to N = T
        self.x_pos_ref: np.ndarray = np.empty(0)
        self.y_pos_ref: np.ndarray = np.empty(0)
        self.z_pos_ref: np.ndarray = np.empty(0)
        self.x_vel_ref: np.ndarray = np.empty(0)
        self.y_vel_ref: np.ndarray = np.empty(0)
        self.z_vel_ref: np.ndarray = np.empty(0)
        self.roll_ref: np.ndarray = np.empty(0)
        self.pitch_ref: np.ndarray = np.empty(0)
        self.yaw_ref: np.ndarray = np.empty(0)
        self.roll_rate_ref: np.ndarray = np.empty(0)
        self.pitch_rate_ref: np.ndarray = np.empty(0)
        self.yaw_rate_ref: np.ndarray = np.empty(0)

        self.dummy_go2 = PinGo2Model()


        self.x_pos_des: np.ndarray = 0
        self.y_pos_des: np.ndarray = 0
        self.pitch_init = 0
        self.roll_init = 0

    def compute_x_ref_vec(self):
        refs = [
            self.x_pos_ref,
            self.y_pos_ref,
            self.z_pos_ref,
            self.roll_ref,
            self.pitch_ref,
            self.yaw_ref,
            self.x_vel_ref,
            self.y_vel_ref,
            self.z_vel_ref,
            self.roll_rate_ref,
            self.pitch_rate_ref,
            self.yaw_rate_ref,
        ]
        # stack into shape (12, N)
        N = min(len(r) for r in refs)
        ref_traj = np.vstack([r[:N] for r in refs])
        return ref_traj

    # def raibertFootPlacement(self, p_com, x_vel, y_vel, frequency, duty):
    #     period = 1/frequency
    #     stanceTime = duty * period
    #     swingTime = (1-duty) * period

    #     p_com[2] = 0

    #     r_next_touchdown_world = p_com + np.array([x_vel*(swingTime + stanceTime/2), y_vel*(swingTime + stanceTime/2), 0])
    #     return r_next_touchdown_world

    def generateConstantTraj(self,
                             go2: PinGo2Model,
                             gait: Gait,
                             time_now: float,
                             x_vel_des: float,
                             y_vel_des: float,
                             z_pos_des: float,
                             yaw_rate_des: float,
                             time_step: float,
                             time_horizon: float):
        


        self.initial_x_vec= go2.compute_com_x_vec()
        initial_pos = self.initial_x_vec[0:3]

        max_pos_error = 0.1   # define the threshold for error of position
        
        if self.x_pos_des - initial_pos[0]> max_pos_error:
            self.x_pos_des = initial_pos[0] + max_pos_error
        if initial_pos[0] - self.x_pos_des > max_pos_error:
            self.x_pos_des = initial_pos[0] - max_pos_error

        if self.y_pos_des - initial_pos[1] > max_pos_error:
            self.y_pos_des = initial_pos[1] + max_pos_error
        if initial_pos[1] - self.y_pos_des > max_pos_error:
            self.y_pos_des = initial_pos[1] - max_pos_error

        go2.x_vel_des =x_vel_des
        go2.y_vel_des =y_vel_des

        go2.x_pos_des =float(self.x_pos_des)
        go2.y_pos_des =float(self.y_pos_des)
        
        self.N = int(time_horizon / time_step) # number of sequences to output
        N = self.N
        t_vec = np.arange(N) * time_step # time vector
        t_vec = t_vec + time_step

        self.time = t_vec
        self.x_pos_ref = self.x_pos_des + x_vel_des * t_vec
        self.x_pos_des = self.x_pos_ref[0]
        self.x_vel_ref = np.full(N, x_vel_des, dtype=float)

        self.y_pos_ref = self.y_pos_des + y_vel_des * t_vec
        self.y_pos_des = self.y_pos_ref[0]
        self.y_vel_ref = np.full(N, y_vel_des, dtype=float)


        self.z_pos_ref = np.full(N, z_pos_des, dtype=float)
        self.z_vel_ref = np.full(N, 0, dtype=float)

        
        [yaw, _, _] = go2.current_config.compute_euler_angle()
        yaw = 0
        self.yaw_ref = yaw + yaw_rate_des * t_vec
        self.yaw_rate_ref = np.full(N, yaw_rate_des, dtype=float)

        if np.fabs(self.initial_x_vec[5]) > 0.2:
            self.roll_init += time_step * (0.0 - self.initial_x_vec[3]) / self.initial_x_vec[5]

        if np.fabs(self.initial_x_vec[6]) > 0.2:
            self.pitch_init += time_step * (0.0 - self.initial_x_vec[4]) / self.initial_x_vec[6]

        # clamp the learned gains
        self.roll_init  = np.fmin(np.fmax(self.roll_init,  -0.25), 0.25)
        self.pitch_init = np.fmin(np.fmax(self.pitch_init, -0.25), 0.25)

        roll_comp  = self.initial_x_vec[5] * self.roll_init   # vy * roll_init
        pitch_comp = self.initial_x_vec[6]  * self.pitch_init  # vx * pitch_init

        self.pitch_ref = np.full(N, pitch_comp, dtype=float)
        self.pitch_rate_ref = np.full(N, 0, dtype=float)

        self.roll_ref = np.full(N, roll_comp, dtype=float)
        self.roll_rate_ref = np.full(N, 0, dtype=float)

        self.contact_table = gait.compute_contact_table(time_now, time_step, N)

        r_fl_traj_world = np.zeros((3,N))
        r_fr_traj_world = np.zeros((3,N))
        r_rl_traj_world = np.zeros((3,N))
        r_rr_traj_world = np.zeros((3,N))




        [r_fl_next_td_world, r_fr_next_td_world, r_rl_next_td_world, r_rr_next_td_world] = go2.get_foot_lever_world()

        mask_previous = np.array([2,2,2,2])
        start = time.perf_counter()
        q = np.zeros(6)
        dq = np.zeros(6)


        for i in range(N):
            current_mask = gait.compute_current_mask(time_now + i * time_step)



            q[0:3] = [self.x_pos_ref[i], self.y_pos_ref[i], self.z_pos_ref[i]]
            q[3:6] = [self.roll_ref[i], self.pitch_ref[i], self.yaw_ref[i]]
            dq[0:3] = [self.x_vel_ref[i], self.y_vel_ref[i], self.z_vel_ref[i]]
            dq[3:6] = [self.roll_rate_ref[i], self.pitch_rate_ref[i], self.yaw_rate_ref[i]]
            self.dummy_go2.update_model_simplified(q, dq)

            self.dummy_go2.x_pos_des

            p_base_traj_world = self.dummy_go2.current_config.base_pos

            ## Front-left foot
            if current_mask[0] != mask_previous[0] and current_mask[0] == 0:
                # Takes off
                pos_fl_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "FL") # This returns the next touchdown position in world coordinate
                r_fl_next_td_world = pos_fl_next_td_world - p_base_traj_world

                r_fl_traj_world[:,i] = np.array([0,0,0])

            if current_mask[0] != mask_previous[0] and current_mask[0] == 1:
                # Touch down
                r_fl_traj_world[:,i] = r_fl_next_td_world # Update the touchdown position 

            if current_mask[0] == mask_previous[0]:
                # No change from last time step
                r_fl_traj_world[:,i] = r_fl_traj_world[:,i-1] # No change, reuse last value 


            ## Front-right foot
            if current_mask[1] != mask_previous[1] and current_mask[1] == 0:
                # Takes off
                pos_fr_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "FR") # This returns the next touchdown position in world coordinate
                r_fr_next_td_world = pos_fr_next_td_world - p_base_traj_world

                r_fr_traj_world[:,i] = np.array([0,0,0])

            if current_mask[1] != mask_previous[1] and current_mask[1] == 1:
                # Touch down
                r_fr_traj_world[:,i] = r_fr_next_td_world # Update the touchdown position 

            if current_mask[1] == mask_previous[1]:
                # No change from last time step
                r_fr_traj_world[:,i] = r_fr_traj_world[:,i-1] # No change, reuse last value 

            ## Rear-left foot
            if current_mask[2] != mask_previous[2] and current_mask[2] == 0:
                # Takes off
                pos_rl_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "RL") # This returns the next touchdown position in world coordinate
                r_rl_next_td_world = pos_rl_next_td_world - p_base_traj_world

                r_rl_traj_world[:,i] = np.array([0,0,0])
            elif current_mask[2] != mask_previous[2] and current_mask[2] == 1:
                # Touch down
                r_rl_traj_world[:,i] = r_rl_next_td_world # Update the touchdown position 

            elif current_mask[2] == mask_previous[2]:
                # No change from last time step
                r_rl_traj_world[:,i] = r_rl_traj_world[:,i-1] # No change, reuse last value 


            ## Rear-right foot
            if current_mask[3] != mask_previous[3] and current_mask[3] == 0:
                # Takes off
                pos_rr_next_td_world = gait.compute_touchdown_world_for_traj_purpose_only(self.dummy_go2, "RR") # This returns the next touchdown position in world coordinate
                r_rr_next_td_world = pos_rr_next_td_world - p_base_traj_world

                r_rr_traj_world[:,i] = np.array([0,0,0])
            elif current_mask[3] != mask_previous[3] and current_mask[3] == 1:
                # Touch down
                r_rr_traj_world[:,i] = r_rr_next_td_world # Update the touchdown position 

            elif current_mask[3] == mask_previous[3]:
                # No change from last time step
                r_rr_traj_world[:,i] = r_rr_traj_world[:,i-1] # No change, reuse last value 


            mask_previous = current_mask

        # Save
        self.r_fl_foot_world = r_fl_traj_world
        self.r_fr_foot_world = r_fr_traj_world
        self.r_rl_foot_world = r_rl_traj_world
        self.r_rr_foot_world = r_rr_traj_world