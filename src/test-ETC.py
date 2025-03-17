import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Simulating the control sequences (MPC output)
def generate_control_sequence(T, H, base_values=(0, 0)):
    """
    Generate a sequence of control inputs over T time steps, each with H+1 control actions.
    The control action is 2D (e.g., surge, sway).
    """
    control_sequence = []
    for t in range(T):
        # Generate 2D control signals (e.g., surge, sway)
        surge = base_values[0] + np.random.uniform(-1, 1)
        sway = base_values[1] + np.random.uniform(-1, 1)
        control_sequence.append([surge, sway])  # Each control point is 2D
    return np.array(control_sequence)

# Compute DTW (Dynamic Time Warping) distance between two 2D control sequences
def compute_dtw(U_k, U_k1):
    """
    Compute the DTW distance between two control sequences U_k and U_k1.
    Each sequence is a 2D array (time steps with surge and sway).
    """
    # Ensure U_k and U_k1 are 2D arrays
    U_k = np.array(U_k).reshape(-1, 2)  # Ensure each control point is 2D
    U_k1 = np.array(U_k1).reshape(-1, 2)

    # Convert the control sequences into a list of tuples for DTW
    U_k_tuples = [tuple(x) for x in U_k]  # Each control point is a 2D tuple
    U_k1_tuples = [tuple(x) for x in U_k1]
    
    distance, _ = fastdtw(U_k_tuples, U_k1_tuples, dist=euclidean)
    return distance

# Decide whether to retransmit based on the DTW distance and adaptive threshold
def retransmit_decision(U_k, U_k1, base_threshold, packet_loss_factor, latency_factor, consensus_error, alpha):
    """
    Decide whether to retransmit based on the discrepancy between control sequences (DTW) 
    and the adaptive threshold considering packet loss, latency, and consensus error.
    """
    # Compute the DTW distance between U_k and U_k1
    dtw_distance = compute_dtw(U_k, U_k1)
    
    # Update the adaptive threshold based on consensus error
    adaptive_threshold = base_threshold - alpha * consensus_error

    retransmit = dtw_distance >= adaptive_threshold
    return retransmit, adaptive_threshold, dtw_distance

# Simulate different consensus behaviors
def consensus_error_behavior(T, behavior_type):
    """
    Simulate different behaviors for the consensus error.
    'sinusoidal' -> Sinusoidal behavior
    'random' -> Random fluctuation
    'converge' -> Converging to zero over time
    """
    if behavior_type == 'sinusoidal':
        consensus_error = np.sin(np.linspace(0, 2 * np.pi, T))  # Sinusoidal error
    elif behavior_type == 'random':
        consensus_error = np.random.uniform(0, 1, T)  # Random error
    elif behavior_type == 'converge':
        consensus_error = np.exp(-np.linspace(0, 5, T))  # Error converges to zero
    else:
        raise ValueError("Invalid behavior type. Choose from 'sinusoidal', 'random', or 'converge'.")
    
    return consensus_error

# Simulate bandwidth usage and compare methods
def compare_bandwidth(T, H, base_threshold, packet_loss_factor, latency_factor, consensus_behavior, alpha, base_message_size=100):
    """
    Compare the bandwidth usage between the periodic and proposed communication methods, with consensus error affecting the threshold.
    """
    periodic_bandwidth = 0
    proposed_bandwidth = 0
    periods_sent = 0

    # Lists for plotting DTW distance, adaptive threshold, and consensus error over time
    dtw_distances = []
    adaptive_thresholds = []
    consensus_errors = []
    retransmissions = []
    divergences = []  # Track divergence between U_k and U_k+1

    # Generate control sequences for T periods, each with H+1 steps
    U_k = generate_control_sequence(T, H)
    
    # Simulate the consensus error behavior
    consensus_error = consensus_error_behavior(T, consensus_behavior)

    for t in range(T - 1):
        U_k_current = U_k[t]
        U_k1_current = U_k[t + 1]
        
        # Periodic communication always sends data
        periodic_bandwidth += base_message_size
        periods_sent += 1
        
        # Proposed communication only sends if retransmission is needed
        retransmit, adaptive_threshold, dtw_distance = retransmit_decision(U_k_current, U_k1_current, base_threshold, packet_loss_factor, latency_factor, consensus_error[t], alpha)
        if retransmit:
            proposed_bandwidth += base_message_size
        else:
            proposed_bandwidth += 0  # No transmission if no retransmission is needed
        
        # Record data for plotting
        dtw_distances.append(dtw_distance)
        adaptive_thresholds.append(adaptive_threshold)
        consensus_errors.append(consensus_error[t])
        retransmissions.append(retransmit)

        # Track divergence (difference between U_k and U_k+1)
        divergence = np.linalg.norm(U_k_current - U_k1_current)
        divergences.append(divergence)

        '''# Reset divergence to zero after transmission
        if retransmit:
            divergences[-1] = 0'''

    # Plot bandwidth comparison
    plt.figure(figsize=(10, 6))
    labels = ['Proposed Method', 'Periodic Communication']
    bandwidths = [proposed_bandwidth, periodic_bandwidth]
    plt.bar(labels, bandwidths, color=['blue', 'green'])
    plt.ylabel('Bandwidth Occupation (bits)')
    plt.title(f'Bandwidth Comparison: Proposed vs Periodic\nPacket Loss Factor: {packet_loss_factor}, Latency Factor: {latency_factor}')
    plt.show()

    # Plot DTW Distance vs. Adaptive Threshold with retransmissions
    plt.figure(figsize=(10, 6))
    plt.plot(dtw_distances, label='DTW Distance', color='blue', linestyle='-', marker='o')
    plt.plot(adaptive_thresholds, label='Adaptive Threshold', color='red', linestyle='--')
    
    # Highlight retransmissions
    retransmit_times = [i for i, retransmit in enumerate(retransmissions) if retransmit]
    plt.scatter(retransmit_times, [dtw_distances[i] for i in retransmit_times], color='green', label='Retransmissions', zorder=5)

    plt.xlabel('Time Steps')
    plt.ylabel('Distance / Threshold')
    plt.title('DTW Distance vs Adaptive Threshold\nwith Retransmissions Highlighted')
    plt.legend()
    plt.show()

    # Plot Adaptive Threshold and Consensus Error over time
    plt.figure(figsize=(10, 6))
    plt.plot(consensus_errors, label='Consensus Error', color='orange', linestyle='-.')
    plt.plot(adaptive_thresholds, label='Adaptive Threshold', color='red', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Consensus Error / Threshold')
    plt.title(f'Adaptive Threshold vs Consensus Error\nfor Behavior: {consensus_behavior}')
    plt.legend()
    plt.show()

    # Plot Divergence vs Adaptive Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(divergences, label='Divergence (U_k vs U_k+1)', color='purple', linestyle='-', marker='x')
    plt.plot(adaptive_thresholds, label='Adaptive Threshold', color='red', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Divergence / Threshold')
    plt.title(f'Divergence vs Adaptive Threshold\nfor Behavior: {consensus_behavior}')
    plt.legend()
    plt.show()

    # Print bandwidth values
    print(f"Proposed Method Bandwidth: {proposed_bandwidth} bits")
    print(f"Periodic Communication Bandwidth: {periodic_bandwidth} bits")

# Run simulation with different packet loss and latency factors
def run_simulation():
    T = 50  # Number of time steps
    H = 5   # Horizon for the MPC (control sequence length)
    base_threshold = 1.5  # Base threshold for retransmission
    packet_loss_factor = 0.2  # Packet loss factor (20%)
    latency_factor = 0.1  # Latency factor (10%)
    alpha = 1.0  # Influence of consensus error on threshold

    consensus_behavior = 'converge'  # Change to 'random' or 'converge' for different behaviors
    compare_bandwidth(T, H, base_threshold, packet_loss_factor, latency_factor, consensus_behavior, alpha)

if __name__ == "__main__":
    run_simulation()
