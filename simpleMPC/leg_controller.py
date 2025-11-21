import numpy as np
from go2_robot_data import PinGo2Model
from gait import Gait

class LegController():
        
        def __init__(self):

            Kp_swing = np.diag([500, 500, 500])
            Kd_swing = np.diag([200, 200, 200])
            self.Kp_swing = Kp_swing
            self.Kd_swing = Kd_swing
            self.last_mask = np.array([2, 2, 2, 2])


        def compute_FL_torque(self, go2: PinGo2Model, gait: Gait, 
                              contact_force: np.ndarray, current_time: float, dt):
              
            J_foot_world = go2.compute_3x3_foot_Jacobian_world("FL") # (3x3)
            J_full_foot_world = go2.compute_full_foot_Jacobian_world("FL") # (3x18)
            [g, C, M] = go2.compute_dynamcis_terms()

            current_mask = gait.compute_current_mask(current_time)
            tau_cmd = np.zeros([3,1])

            [foot_pos_des, foot_vel_des] = go2.get_single_foot_state_in_world("FL")
            [foot_pos_now, foot_vel_now] = go2.get_single_foot_state_in_world("FL")

            if self.last_mask[0] != current_mask[0] and current_mask[0] == 0:
                # LF foot takes off from the ground
                self.FL_takeoff_time = current_time
                [self.FL_traj, self.FL_td_pos] = gait.compute_swing_traj_and_touchdown(go2, "FL")

            if current_mask[0] == 0: ## Swing
                time_since_takeoff = current_time - self.FL_takeoff_time
                [foot_pos_des, foot_vel_des, foot_acl_des] = self.FL_traj(time_since_takeoff)
                [foot_pos_now, foot_vel_now] = go2.get_single_foot_state_in_world("FL")

                pos_error = (foot_pos_des - foot_pos_now)
                vel_error = (foot_vel_des - foot_vel_now)
                Lambda = np.linalg.inv(J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T) # (3x3)
                Jdot_dq = go2.compute_Jdot_dq_world("FL")

                f_ff = Lambda @ (foot_acl_des - Jdot_dq) # Feedforward term (3x1)

                force = self.Kp_swing @ pos_error + self.Kd_swing @ vel_error + f_ff # PD force (3x1)
                tau_cmd= J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[6:9]
                
            else: ## Stance
                tau_cmd= J_foot_world.T @ -contact_force

            self.last_mask[0] = current_mask[0]

            return tau_cmd, foot_pos_des, foot_pos_now, foot_vel_des, foot_vel_now
        
        def compute_FR_torque(self, go2: PinGo2Model, gait: Gait, 
                              contact_force: np.ndarray, current_time: float, dt):
              
            J_foot_world = go2.compute_3x3_foot_Jacobian_world("FR")
            J_full_foot_world = go2.compute_full_foot_Jacobian_world("FR") # (3x18)
            [g, C, M] = go2.compute_dynamcis_terms()

            current_mask = gait.compute_current_mask(current_time)
            tau_cmd = np.zeros([3,1])

            if self.last_mask[1] != current_mask[1] and current_mask[1] == 0:
                # LF foot takes off from the ground
                self.FR_takeoff_time = current_time
                [self.FR_traj, self.FR_td_pos] = gait.compute_swing_traj_and_touchdown(go2, "FR")

            if current_mask[1] == 0: ## Swing
                time_since_takeoff = current_time - self.FR_takeoff_time
                [foot_pos_des, foot_vel_des, foot_acl_des] = self.FR_traj(time_since_takeoff)
                [foot_pos_now, foot_vel_now] = go2.get_single_foot_state_in_world("FR")

                pos_error = foot_pos_des - foot_pos_now
                vel_error = foot_vel_des - foot_vel_now
                Lambda = np.linalg.inv(J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T) # (3x3)
                Jdot_dq = go2.compute_Jdot_dq_world("FR")

                f_ff = Lambda @ (foot_acl_des - Jdot_dq) # Feedforward term (3x1)

                force = self.Kp_swing @ pos_error + self.Kd_swing @ vel_error + f_ff # PD force (3x1)
                tau_cmd= J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[9:12]
                
            else: ## Stance
                tau_cmd= J_foot_world.T @ -contact_force

            self.last_mask[1] = current_mask[1]

            return tau_cmd
        

        def compute_RL_torque(self, go2: PinGo2Model, gait: Gait, 
                              contact_force: np.ndarray, current_time: float, dt):
              
            J_foot_world = go2.compute_3x3_foot_Jacobian_world("RL")
            J_full_foot_world = go2.compute_full_foot_Jacobian_world("RL") # (3x18)
            [g, C, M] = go2.compute_dynamcis_terms()

            current_mask = gait.compute_current_mask(current_time)
            tau_cmd = np.zeros([3,1])

            if self.last_mask[2] != current_mask[2] and current_mask[2] == 0:
                # LF foot takes off from the ground
                self.RL_takeoff_time = current_time
                [self.RL_traj, self.RL_td_pos] = gait.compute_swing_traj_and_touchdown(go2, "RL")

            if current_mask[2] == 0: ## Swing
                time_since_takeoff = current_time - self.RL_takeoff_time
                [foot_pos_des, foot_vel_des, foot_acl_des] = self.RL_traj(time_since_takeoff)
                [foot_pos_now, foot_vel_now] = go2.get_single_foot_state_in_world("RL")

                pos_error = foot_pos_des - foot_pos_now
                vel_error = foot_vel_des - foot_vel_now
                Lambda = np.linalg.inv(J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T) # (3x3)
                Jdot_dq = go2.compute_Jdot_dq_world("RL")

                f_ff = Lambda @ (foot_acl_des - Jdot_dq) # Feedforward term (3x1)

                force = self.Kp_swing @ pos_error + self.Kd_swing @ vel_error + f_ff # PD force (3x1)
                tau_cmd= J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[12:15]
                
            else: ## Stance
                tau_cmd= J_foot_world.T @ -contact_force

            self.last_mask[2] = current_mask[2]

            return tau_cmd
        
        def compute_RR_torque(self, go2: PinGo2Model, gait: Gait, 
                              contact_force: np.ndarray, current_time: float, dt):
              
            J_foot_world = go2.compute_3x3_foot_Jacobian_world("RR")
            J_full_foot_world = go2.compute_full_foot_Jacobian_world("RR") # (3x18)
            [g, C, M] = go2.compute_dynamcis_terms()

            current_mask = gait.compute_current_mask(current_time)
            tau_cmd = np.zeros([3,1])

            if self.last_mask[3] != current_mask[3] and current_mask[3] == 0:
                # LF foot takes off from the ground
                self.RR_takeoff_time = current_time
                [self.RR_traj, self.RR_td_pos] = gait.compute_swing_traj_and_touchdown(go2, "RR")

            if current_mask[3] == 0: ## Swing
                time_since_takeoff = current_time - self.RR_takeoff_time
                [foot_pos_des, foot_vel_des, foot_acl_des] = self.RR_traj(time_since_takeoff)
                [foot_pos_now, foot_vel_now] = go2.get_single_foot_state_in_world("RR")

                pos_error = foot_pos_des - foot_pos_now
                vel_error = foot_vel_des - foot_vel_now
                Lambda = np.linalg.inv(J_full_foot_world @ np.linalg.inv(M) @ J_full_foot_world.T) # (3x3)
                Jdot_dq = go2.compute_Jdot_dq_world("RR")

                f_ff = Lambda @ (foot_acl_des - Jdot_dq) # Feedforward term (3x1)

                force = self.Kp_swing @ pos_error + self.Kd_swing @ vel_error + f_ff # PD force (3x1)
                tau_cmd= J_foot_world.T @ force + (C @ go2.current_config.get_dq() + g)[15:18]
                
            else: ## Stance
                tau_cmd= J_foot_world.T @ -contact_force

            self.last_mask[3] = current_mask[3]

            return tau_cmd
              
              
