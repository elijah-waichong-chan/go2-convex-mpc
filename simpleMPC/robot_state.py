import numpy as np
import pinocchio as pin

class ConfigurationState:

    def __init__(self):

        # Initial generalized positions
        self.base_pos = np.array([0.0, 0.0, 0.27])
        self.base_quad = np.array([0.0, 0.0, 0.0, 1.0])
        self.FL_joint_angle = np.array([0.0, 0.9, -1.8])
        self.FR_joint_angle =  np.array([0.0, 0.9, -1.8])
        self.RL_joint_angle = np.array([0.0, 0.9, -1.8])
        self.RR_joint_angle = np.array([0.0, 0.9, -1.8])

        # Initial generalized velocities
        self.base_vel = np.array([0.0, 0.0, 0.0])
        self.base_ang_vel = np.array([0.0, 0.0, 0.0])
        self.FL_joint_vel = np.array([0.0, 0.0, 0.0])
        self.FR_joint_vel = np.array([0.0, 0.0, 0.0])
        self.RL_joint_vel = np.array([0.0, 0.0, 0.0])
        self.RR_joint_vel = np.array([0.0, 0.0, 0.0])



    def get_q(self):
        #Generalized position: (19x1)
        q = np.concatenate([self.base_pos, self.base_quad, 
                            self.FL_joint_angle, self.FR_joint_angle,
                            self.RL_joint_angle, self.RR_joint_angle])
        return q
    
    def get_dq(self):
        #Generalized velocity: (18x1)
        dq = np.concatenate([self.base_vel, self.base_ang_vel, 
                            self.FL_joint_vel, self.FR_joint_vel,
                            self.RL_joint_vel, self.RR_joint_vel])
        return dq
    
    def update_q(self, q):
        # base pose
        self.base_pos  = q[0:3]  # [x, y, z]
        self.base_quad = q[3:7]  # quaternion [x, y, z, w]

        # joint angles: FL, FR, RL, RR each [hip, thigh, calf]
        j = q[7:19]
        self.FL_joint_angle = j[0:3]
        self.FR_joint_angle = j[3:6]
        self.RL_joint_angle = j[6:9]
        self.RR_joint_angle = j[9:12]

    def update_dq(self, v):

        # base twist
        self.base_vel     = v[0:3]      # [vx, vy, vz]
        self.base_ang_vel = v[3:6]      # [wx, wy, wz]

        # joint velocities: FL, FR, RL, RR each [hip, thigh, calf]
        jv = v[6:18]
        self.FL_joint_vel = jv[0:3]
        self.FR_joint_vel = jv[3:6]
        self.RL_joint_vel = jv[6:9]
        self.RR_joint_vel = jv[9:12]
    
    def compute_euler_angle(self):

        q_eig = pin.Quaternion(self.base_quad)
        R = q_eig.toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(R)

        return np.array(rpy).reshape(3,)
    
    def update_with_euler_angle(self, roll, pitch, yaw):

        cr,sr = np.cos(roll/2), np.sin(roll/2)
        cp,sp = np.cos(pitch/2), np.sin(pitch/2)
        cy,sy = np.cos(yaw/2), np.sin(yaw/2)
        
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy
        qw = cr*cp*cy + sr*sp*sy

        self.base_quad = np.array([qx, qy, qz, qw])
    