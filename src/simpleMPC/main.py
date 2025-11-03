from go2_model import Pin_Go2_Model
from trajectoryPlanner import RigidBodyTraj
from dynamics import Dynamics
from mpc_opt import build_mpc
import time
from plot_helper import plot_contact_forces, plot_traj_tracking
import numpy as np

# 2) Compute the dynamics and desired trajectory
go2 = Pin_Go2_Model()
traj = RigidBodyTraj()
dynamics = Dynamics()
dt = 0.04

traj.generateConstantTraj(go2, x_vel_des=0.5, y_vel_des=0.2, z_pos_des=0.27, yaw_rate_des=0, t0=0, time_step=dt, time_horizon=1, frequency=1, duty=0.5)
dynamics.continuousDynamics(go2, traj)
dynamics.discreteDynamics(dt)



# solver, assemble_qp, sizes = build_matrix_mpc(dynamics, traj, mu=0.7, fz_min=10.0)

# x0 = go2.current_config.get_simplified_full_state().compute_x_vec()
# x_ref_traj = traj.compute_x_ref_vec()
# H, gvec, A, lba, uba, lbx, ubx = assemble_qp(x0, x_ref_traj, dynamics.Bd)
# sol = solver(h=H, g=gvec, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)


# 2) Build the Optimization Program
t0 = time.perf_counter()
solver, sizes, bounds = build_mpc(dynamics, go2, traj)
t1 = time.perf_counter()
t_build = t1 - t0
print(f"QP Build Time: {t_build*1e3:.3f} ms")

# 3) Solve
t0 = time.perf_counter()
sol = solver(lbx=bounds['lbx'],
             ubx=bounds['ubx'],
             lbg=bounds['lbg'],
             ubg=bounds['ubg'])
t1 = time.perf_counter()
t_solve = t1 - t0
print(f"QP solve time: {t_solve*1e3:.3f} ms ({1.0/t_solve:.1f} Hz equivalent)")

# 4) Retrieve results
nx, nu, N = sizes['nx'], sizes['nu'], sizes['N']
w_opt = sol["x"].full().flatten()  # (n_w,)
X_opt = w_opt[: nx*(N+1)].reshape((nx, N+1), order='F')
U_opt = w_opt[nx*(N+1):].reshape((nu, N), order='F')

print(np.shape(w_opt))

# 5) Run simulation with optimal input
[x_now, x_traj] = dynamics.run_simulation(go2, U_opt)

pos_traj_sim = x_traj[0:3, :]
x0_col = go2.current_config.get_simplified_full_state().compute_x_vec().reshape(-1, 1)   # (nx,1)
pos_traj_ref = np.hstack([x0_col[0:3, :], traj.compute_x_ref_vec()[0:3, :]])

plot_traj_tracking(pos_traj_ref, pos_traj_sim, block=False)
plot_contact_forces(U_opt, traj.contact_schedule, dt, block=True)
