import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
N_AUV = 3               # Number of agents
H = 10                  # Prediction horizon
SIM_TIME = 100          # Simulation steps
dt = 1.0                # Time step (sec)

# Target moves in a circle
def target_trajectory(t):
    R = 50
    omega = 0.05
    x = R * np.cos(omega * t)
    y = R * np.sin(omega * t)
    return np.array([x, y])

# Simple AUV motion model: position update (x,y)
def move_auv(pos, vel, dt):
    return pos + vel * dt

# Generate MPC predicted sequence: assume AUV plans a straight line to target over horizon
def generate_predicted_states(current_pos, target_pos, H):
    seq = []
    for h in range(1, H+1):
        alpha = h / H
        pred = (1 - alpha) * current_pos + alpha * target_pos
        seq.append(pred)
    return np.array(seq)  # shape (H, 2)

# Weighted distance metric (ETC trigger)
def weighted_distance(seq1, seq2, alpha=0.8):
    w = np.array([alpha**(h) for h in range(len(seq1))])
    dist = np.linalg.norm(seq1 - seq2, axis=1)
    return np.sum(w * dist)

# Alternative trigger: simple max norm difference over horizon
def max_norm_distance(seq1, seq2):
    dist = np.linalg.norm(seq1 - seq2, axis=1)
    return np.max(dist)

# Simulation function
def run_simulation(beta, lambd, delta_min, trigger_method='weighted', alpha_w=0.8):
    # Initialize agent positions randomly in a circle radius 30
    auv_pos = np.array([ [30*np.cos(2*np.pi*i/N_AUV), 30*np.sin(2*np.pi*i/N_AUV)] for i in range(N_AUV) ])
    auv_prev_pred = [generate_predicted_states(pos, target_trajectory(0), H) for pos in auv_pos]
    k_last_comm = np.zeros(N_AUV)  # last comm step per agent

    comm_count = np.zeros(N_AUV)  # count communications
    tracking_errors = np.zeros((SIM_TIME, N_AUV))

    for k in range(1, SIM_TIME+1):
        t = k*dt
        target_pos = target_trajectory(t)

        for i in range(N_AUV):
            # Generate new prediction from current position to target
            pred_seq = generate_predicted_states(auv_pos[i], target_pos, H)

            # Calculate trigger metric
            if trigger_method == 'weighted':
                dist = weighted_distance(pred_seq, auv_prev_pred[i], alpha=alpha_w)
            elif trigger_method == 'maxnorm':
                dist = max_norm_distance(pred_seq, auv_prev_pred[i])
            else:
                raise ValueError("Unknown trigger method")

            # Adaptive threshold
            delta_k = max(delta_min, beta * (lambd**(k - k_last_comm[i])) )

            # Trigger check
            if dist > delta_k:
                # Transmit: update previous prediction and reset timer
                auv_prev_pred[i] = pred_seq
                k_last_comm[i] = k
                comm_count[i] += 1

            # Move AUV one step toward the first predicted point in the sequence
            direction = pred_seq[0] - auv_pos[i]
            speed = np.linalg.norm(direction)
            if speed > 0:
                vel = direction / speed * min(speed, 1.0)  # max speed = 1.0 unit/s
            else:
                vel = np.zeros(2)
            auv_pos[i] = move_auv(auv_pos[i], vel, dt)

            # Tracking error (distance from target)
            tracking_errors[k-1,i] = np.linalg.norm(auv_pos[i] - target_pos)

    avg_comm = np.mean(comm_count)
    avg_error = np.mean(tracking_errors[-10:,:])  # last 10 steps average error

    return avg_comm, avg_error, comm_count, tracking_errors

# Sensitivity analysis over beta and lambda
def sensitivity_analysis():
    betas = [1.0, 2.0, 5.0, 10.0]
    lambdas = [0.95, 0.9, 0.85, 0.8]
    delta_min = 0.1

    results_weighted = []
    results_maxnorm = []

    for beta in betas:
        for lambd in lambdas:
            # Weighted trigger
            comm_w, err_w, _, _ = run_simulation(beta, lambd, delta_min, 'weighted')
            results_weighted.append((beta, lambd, comm_w, err_w))

            # Maxnorm trigger
            comm_m, err_m, _, _ = run_simulation(beta, lambd, delta_min, 'maxnorm')
            results_maxnorm.append((beta, lambd, comm_m, err_m))

    return results_weighted, results_maxnorm

# Plotting tradeoff
def plot_results(results, title):
    plt.figure(figsize=(8,6))
    for beta in sorted(set(r[0] for r in results)):
        subset = [r for r in results if r[0] == beta]
        lambdas = [r[1] for r in subset]
        comms = [r[2] for r in subset]
        errs = [r[3] for r in subset]
        plt.plot(comms, errs, marker='o', label=f'beta={beta}')
    plt.xlabel('Average Number of Communications per AUV')
    plt.ylabel('Average Tracking Error (last 10 steps)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Run and plot
if __name__ == "__main__":
    results_w, results_m = sensitivity_analysis()
    plot_results(results_w, "Trade-off for Weighted Distance ETC")
    plot_results(results_m, "Trade-off for Max-Norm Distance ETC")
