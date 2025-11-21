from pinocchio.robot_wrapper import RobotWrapper
from pathlib import Path
import pinocchio as pin 
from robot_state import ConfigurationState
import numpy as np
from numpy import sin, cos
from scipy.signal import cont2discrete
from scipy.linalg import expm

class PinGo2Model:

    def __init__(self):
        # Locate URDF relative to this file
        repo = Path(__file__).resolve().parents[1]
        urdf_path = repo / "go2_description" / "urdf" / "go2_description.urdf"

        # Build robot (free-flyer at the root)
        robot = RobotWrapper.BuildFromURDF(
            str(urdf_path),
            package_dirs=[str(repo)],
            root_joint=pin.JointModelFreeFlyer()
        )

        # Core models
        self.model = robot.model
        self.vmodel = robot.visual_model
        self.cmodel = robot.collision_model
        # Initial data containers
        self.data = self.model.createData()

        # Initial configuration
        self.current_config = ConfigurationState()
        self.q_init = self.current_config.get_q()
        self.dq_init = self.current_config.get_dq()

        # Forward kinematics / frame placements at q_init
        pin.forwardKinematics(self.model, self.data, self.q_init)
        pin.updateFramePlacements(self.model, self.data)

        self.base_id = self.model.getFrameId("base")

        self.FL_foot_id = self.model.getFrameId("FL_foot_joint")
        self.FR_foot_id = self.model.getFrameId("FR_foot_joint")
        self.RL_foot_id = self.model.getFrameId("RL_foot_joint")
        self.RR_foot_id = self.model.getFrameId("RR_foot_joint")

        self.FL_hip_id = self.model.getFrameId("FL_thigh_joint")
        self.FR_hip_id = self.model.getFrameId("FR_thigh_joint")
        self.RL_hip_id = self.model.getFrameId("RL_thigh_joint")
        self.RR_hip_id = self.model.getFrameId("RR_thigh_joint")

        oMb = self.data.oMf[self.base_id]
        oMh1 = self.data.oMf[self.FL_hip_id]
        oMh2 = self.data.oMf[self.FR_hip_id]
        oMh3 = self.data.oMf[self.RL_hip_id]
        oMh4 = self.data.oMf[self.RR_hip_id]

        bMh1 = oMb.actInv(oMh1)
        bMh2 = oMb.actInv(oMh2)
        bMh3 = oMb.actInv(oMh3)
        bMh4 = oMb.actInv(oMh4)

        self.FL_hip_offset = bMh1.translation.copy()
        self.FR_hip_offset = bMh2.translation.copy()
        self.RL_hip_offset = bMh3.translation.copy()
        self.RR_hip_offset = bMh4.translation.copy()

        self.update_model(self.q_init, self.dq_init)

        # Gravity Vector
        self.gc = np.array([
            0, 0, 0, 
            0, 0, 0, 
            0, 0, -9.81, 
            0, 0, 0 
            ])

        self.x_pos_des = []
        self.y_pos_des = []
        self.x_vel_des = []
        self.y_vel_des = []

    def get_hip_offset(self, leg: str):
        name = f"{leg.upper()}_hip_offset"
        return getattr(self, name)
    
    def compute_com_x_vec(self):

        pos_com_world = self.pos_com_world
        vel_com_world = self.vel_com_world

        x_vec = np.concatenate([pos_com_world, self.current_config.compute_euler_angle(), 
                                vel_com_world, self.current_config.base_ang_vel])
        
        x_vec = x_vec.reshape(-1, 1)

        return x_vec

    def update_model(self, q, dq):
        self.current_config.update_q(q)
        self.current_config.update_dq(dq)
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data) 
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeJointJacobians(self.model,self.data,q)
        pin.ccrba(self.model, self.data, q, dq)
        pin.centerOfMass(self.model, self.data, q) 

        self.oMb = self.data.oMf[self.base_id]
        self.oMf1 = self.data.oMf[self.FL_foot_id]
        self.oMf2 = self.data.oMf[self.FR_foot_id]
        self.oMf3 = self.data.oMf[self.RL_foot_id]
        self.oMf4 = self.data.oMf[self.RR_foot_id]
        self.pos_com_world = self.data.com[0].copy() 
        self.vel_com_world = self.data.vcom[0].copy() 
        
        self.R_base_to_world = self.oMb.rotation
        self.R_world_to_base = self.R_base_to_world.T

    def update_model_simplified(self, q, dq):

        roll = q[3]
        pitch = q[4]
        yaw = q[5]

        cr,sr = np.cos(roll/2), np.sin(roll/2)
        cp,sp = np.cos(pitch/2), np.sin(pitch/2)
        cy,sy = np.cos(yaw/2), np.sin(yaw/2)
        
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy

        q_full = np.concatenate([
            q[0:3],                # base position
            [qx, qy, qz, qw],      # base quaternion
            np.zeros(12)           # 12 leg joint angles
        ])

        dq_full = np.concatenate([
            dq[0:6],              
            np.zeros(12)
        ])

        self.current_config.update_q(q_full)
        self.current_config.update_dq(dq_full)
        pin.forwardKinematics(self.model, self.data, q_full, dq_full)
        pin.updateFramePlacements(self.model, self.data) 
        pin.computeAllTerms(self.model, self.data, q_full, dq_full)
        pin.computeJointJacobians(self.model,self.data, q_full)
        pin.ccrba(self.model, self.data, q_full, dq_full)
        pin.centerOfMass(self.model, self.data, q_full) 

        self.oMb = self.data.oMf[self.base_id]
        self.oMf1 = self.data.oMf[self.FL_foot_id]
        self.oMf2 = self.data.oMf[self.FR_foot_id]
        self.oMf3 = self.data.oMf[self.RL_foot_id]
        self.oMf4 = self.data.oMf[self.RR_foot_id]
        self.pos_com_world = self.data.com[0].copy() 
        self.vel_com_world = self.data.vcom[0].copy() 

        self.R_base_to_world = self.oMb.rotation
        self.R_world_to_base = self.R_base_to_world.T

    def get_foot_placement_in_world(self):

        FL_placement = self.oMf1.translation.copy()
        FR_placement = self.oMf2.translation.copy()
        RL_placement = self.oMf3.translation.copy()
        RR_placement = self.oMf4.translation.copy()

        return FL_placement, FR_placement, RL_placement, RR_placement
    
    def get_foot_lever_world(self):

        pos_com_world = self.pos_com_world    
        FL_placement = self.oMf1.translation - pos_com_world
        FR_placement = self.oMf2.translation - pos_com_world
        RL_placement = self.oMf3.translation - pos_com_world
        RR_placement = self.oMf4.translation - pos_com_world

        return FL_placement, FR_placement, RL_placement, RR_placement
    
    def get_single_foot_state_in_world(self, leg: str):

        foot_id = getattr(self, f"{leg}_foot_id")

        # position in world (assumes updateFramePlacements already called)
        oMf = self.data.oMf[foot_id]
        foot_pos_world = oMf.translation.copy()  # (3,)

        # 6D spatial velocity in LOCAL_WORLD_ALIGNED (axes = world)
        v6 = pin.getFrameVelocity(self.model, self.data, foot_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        foot_vel_world = np.array(v6.linear).copy()  # (3,)

        return foot_pos_world, foot_vel_world
    

    # def compute_3x3_foot_Jacobian_base(self, leg: str):

    #     q = self.current_config.compute_q()
    #     footID = self.model.getFrameId(f"{leg}_foot")

    #     oMb = self.data.oMf[self.base_id]

    #     J_world = pin.computeFrameJacobian(self.model, self.data, q, footID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    #     J_pos_world = J_world[0:3,:]
    #     J_pos_base = oMb.rotation.T @ J_pos_world

    #     joint_ids  = [self.model.getJointId(f"{leg}_hip_joint"), 
    #                   self.model.getJointId(f"{leg}_thigh_joint"), 
    #                   self.model.getJointId(f"{leg}_calf_joint")]

    #     vcols = [self.model.joints[jid].idx_v for jid in joint_ids]

    #     J_leg_pos_base = J_pos_base[:, vcols] 

    #     return J_leg_pos_base
    
    def compute_3x3_foot_Jacobian_world(self, leg: str):

        footID = self.model.getFrameId(f"{leg}_foot")
        J_world = pin.getFrameJacobian(self.model, self.data, footID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos_world = J_world[0:3,:]

        joint_ids  = [self.model.getJointId(f"{leg}_hip_joint"), 
                      self.model.getJointId(f"{leg}_thigh_joint"), 
                      self.model.getJointId(f"{leg}_calf_joint")]

        vcols = [self.model.joints[jid].idx_v for jid in joint_ids]

        J_leg_pos_world = J_pos_world[:, vcols] 

        return J_leg_pos_world
    
    def compute_Jdot_dq_world(self, leg:str):
        footID = self.model.getFrameId(f"{leg}_foot")
        Jdot_dq_6 = pin.getFrameClassicalAcceleration(self.model, self.data, footID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
        # Jdot * dq (3x1)
        Jdot_dq = np.array(Jdot_dq_6).reshape(3,)

        return Jdot_dq

    
    def compute_full_foot_Jacobian_world(self, leg: str):

        footID = self.model.getFrameId(f"{leg}_foot")

        J_world = pin.getFrameJacobian(self.model, self.data, footID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_pos_world = J_world[0:3,:]

        return J_pos_world
    
    
    def compute_dynamcis_terms(self):
        g = self.data.g           # gravity torque term (18 x 1)
        C = self.data.C           # Coriolis matrix (18 x 18)
        M = self.data.M           # joint-space inertia matrix (18 x 18)

        return g, C, M


    
    # def inverse_kinematics(self, leg: str, p_des_H: np.array):
    
    #     q = self.current_config.compute_q()

    #     eps = 0.1
    #     IT_MAX = 1000
    #     step = 0.2
    #     damp = 1e-7

    #     footID = self.model.getFrameId(f"{leg}_foot")

    #     success = False

    #     i = 0
    #     while True:
    #         pin.forwardKinematics(self.model, self.data, q)
    #         pin.updateFramePlacements(self.model, self.data)

    #         oMb = self.data.oMf[self.base_id]
    #         oMf = self.data.oMf[footID]

    #         bMf = oMb.actInv(oMf)
    #         p_now_H = bMf.translation

    #         #print(f"{i}: pos = {p_now_H.T}")

    #         e_pos_H = p_des_H - p_now_H
    #         #print(f"{i}: error = {e_pos_H.T}")

    #         if np.linalg.norm(e_pos_H) < eps:
    #             success = True
    #             break
    #         if i >= IT_MAX:
    #             success = False
    #             break

    #         J_world = pin.computeFrameJacobian(self.model, self.data, q, footID, pin.ReferenceFrame.WORLD)
    #         J_pos_world = J_world[:3,:]
    #         J_pos_base = oMb.rotation.T @ J_pos_world

    #         joint_ids  = [self.model.getJointId(f"{leg}_hip_joint"), 
    #                       self.model.getJointId(f"{leg}_thigh_joint"), 
    #                       self.model.getJointId(f"{leg}_calf_joint")]
            
    #         vcols = [self.model.joints[jid].idx_v for jid in joint_ids]
    #         qcols = [self.model.joints[jid].idx_q for jid in joint_ids]
    #         #print(qcols)

    #         J_leg = J_pos_base[:, vcols] 

    #         H = J_leg.T @ J_leg + (damp**2) * np.eye(J_leg.shape[1])
    #         delta_q_leg = np.linalg.solve(H, J_leg.T @ e_pos_H)

    #         q[qcols] = q[qcols] + step * delta_q_leg

    #         i += 1

    #     if success:
    #         print(f"IK Convergence achieved for {leg} foot!")
    #         #print(f"\nresult: {q.flatten().tolist()}")
    #         #print(f"\nfinal error: {e_pos_H.T}")
    #     else:
    #         print(
    #             "\n"
    #             "Warning: the iterative algorithm has not reached convergence "
    #             "to the desired precision"
    #         )

    #     self.update_model(q)

    def update_dynamics(self, traj, dt):
        self._continuousDynamics(traj)
        self._discreteDynamics(dt)


    def run_simulation(self, u_vec):

        N_input = u_vec.shape[1] # Sequence of input given
        assert N_input == self.dynamics_N, f"Expected {N_input=} to equal {self.dynamics_N=}"

        x_traj = np.zeros((12, N_input+1))
        x_init = self.current_config.compute_simplified_x_vec()
        x_traj[:, [0]] = x_init

        for i in range(N_input):
            u_i   = u_vec[:, i].reshape(-1, 1)
            x_traj[:, i+1] = (self.Ad @ x_traj[:, [i]] + self.Bd[i] @ u_i + self.gd).flatten()

        return x_init, x_traj

    def _skew(self,vector):

        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def _continuousDynamics(self, traj):
        
        m = self.data.Ig.mass
        I_com_world = self.data.Ig.inertia # Get current rotational inertia and freeze for the horizon

        I_inv = np.linalg.inv(I_com_world)
        self.dynamics_N = traj.N

        psi_avg = np.average(traj.yaw_ref)

        [_, _, psi_avg] = self.current_config.compute_euler_angle()
        R_z = np.array([
            [cos(psi_avg), -sin(psi_avg), 0],
            [sin(psi_avg),  cos(psi_avg), 0],
            [0,             0,            1]
        ])

        self.Ac = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3),        np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), R_z.T           ],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
        ])

        self.Bc = np.zeros((self.dynamics_N, 12, 12))
        for i in range(self.dynamics_N):

            skew_r1 = self._skew(traj.r_fl_foot_world[:, i])
            skew_r2 = self._skew(traj.r_fr_foot_world[:, i])
            skew_r3 = self._skew(traj.r_rl_foot_world[:, i])
            skew_r4 = self._skew(traj.r_rr_foot_world[:, i])

            self.Bc[i] = np.block([
                [np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3))],
                [np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3)),    np.zeros((3, 3))],
                [(1/m) * np.eye(3), (1/m) * np.eye(3), (1/m) * np.eye(3), (1/m) * np.eye(3)],
                [I_inv @ skew_r1,   I_inv @ skew_r2,   I_inv @ skew_r3,   I_inv @ skew_r4],
            ])

    def _discreteDynamics(self, dt):

        self.Bd = np.zeros((self.dynamics_N, 12, 12))

        # Discretize Ac and Bc
        for i in range(self.dynamics_N): 
            self.Ad, self.Bd[i], *_ = cont2discrete((self.Ac, self.Bc[i], np.eye(12), np.zeros((12, 12))), dt, method='zoh')

        # Discretize gc
        n_steps = 50
        tau = np.linspace(0, dt, n_steps)
        exp_terms = [expm(self.Ac * t) @ self.gc for t in tau]
        gd = np.trapz(np.stack(exp_terms, axis=1), tau, axis=1)

        self.gd = gd.reshape(-1, 1)


