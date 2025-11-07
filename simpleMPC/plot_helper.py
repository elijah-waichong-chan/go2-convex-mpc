import numpy as np
import matplotlib.pyplot as plt

def plot_contact_forces(U_opt, contact_mask, dt, block, leg_names=("FL","FR","RL","RR")):

    assert U_opt.shape[0] == 12, "Expected 12 rows in U_opt (4 legs × 3 forces)."
    N = U_opt.shape[1]
    # print(contact_mask)
    # assert contact_mask.shape == (4, N), "contact_mask must be (4, N)."

    t_edges = np.linspace(0, N*dt, N+1)

    def F(leg_idx):
        base = 3 * leg_idx
        Fleg = U_opt[base:base+3, :]          # (3, N)
        return Fleg[0, :], Fleg[1, :], Fleg[2, :]  # fx, fy, fz each (N,)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, ax in enumerate(axes):
        fx, fy, fz = F(i)

        # Each value held on [t_k, t_{k+1})
        ax.stairs(fx, t_edges, label="fx")
        ax.stairs(fy, t_edges, label="fy")
        ax.stairs(fz, t_edges, label="fz", linewidth=2)

        # Shade swing intervals (mask==0) exactly over each dt-wide bin
        swing = (contact_mask[i] == 0)
        for k in np.flatnonzero(swing):
            ax.axvspan(t_edges[k], t_edges[k+1], alpha=0.15, hatch='//', edgecolor='none')

        ax.set_ylabel(f"{leg_names[i]} [N]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncols=3, fontsize=9)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Leg Contact Forces (one gait cycle)")
    plt.tight_layout()
    plt.show(block=block)   # shows both windows, doesn’t block
    plt.pause(0.001)        # lets the GUI event loop breathe


def plot_traj_tracking(pos_traj_ref, pos_traj_sim, block):
    

    # --- 3D plot ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pos_traj_ref[0,0], pos_traj_ref[1,0], pos_traj_ref[2,0], label="Initial Position")
    ax.plot(pos_traj_ref[0,:], pos_traj_ref[1,:], pos_traj_ref[2,:], 'b--', linewidth=2, label="Reference Trajectory")
    ax.plot(pos_traj_sim[0,:], pos_traj_sim[1,:], pos_traj_sim[2,:], 'g-', linewidth=2, label="Optimal")
    #ax.scatter(pos_traj_sim[0,0], pos_traj_sim[1,0], pos_traj_sim[2,0], label="Optimal")

    ax.set_title("3D Trajectory", fontsize=13)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect([1, 1, 1])

    # --- hard enforce equal data ranges ---
    xs, ys, zs = pos_traj_sim[0,:], pos_traj_sim[1,:], pos_traj_sim[2,:]

    # one symmetric radius for all axes
    r = float(np.max(np.abs([xs, ys, zs])))   # max abs over all axes & time

    if r == 0:   # handle degenerate paths
        r = 1e-6

    xmin, xmax = -r, r
    ymin, ymax = -r, r
    zmin, zmax = -r, r

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.legend()
    ax.grid(True)
    ax.legend(loc="best")
    plt.show(block=block)   # shows both windows, doesn’t block
    plt.pause(0.001)        # lets the GUI event loop breathe

