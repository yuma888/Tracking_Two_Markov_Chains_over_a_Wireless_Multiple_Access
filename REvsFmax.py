import numpy as np
import time
import scipy.io as sio  # For saving .mat files (requires scipy installation: pip install scipy)
import csv  # Backup: Save data in CSV format
from REvsP import *
from DPP import *


# ============================================================
# Monte-Carlo Simulation
# ============================================================
def simulate_policy_mc(cmdp, policy, num_steps=50000, burn_in=2000, count_actions=True):
    """
    Perform Monte-Carlo simulation to evaluate a given stationary randomized policy for reconstruction error (RE).
    Computes long-term average RE, transmission frequencies, and transmission mode statistics.

    Parameters:
        cmdp (TwoSourceMPRCMDP): Instance of the two-source MPR-CMDP model.
        policy (np.ndarray): Randomized policy matrix with shape (S, A), where each row is a probability
                             distribution over actions for the corresponding state.
        num_steps (int): Number of effective simulation steps (excluding burn-in).
        burn_in (int): Burn-in steps to reach steady state (not included in final statistics).
        count_actions (bool): Whether to count transmission mode frequencies (single1/single2/both/none).

    Returns:
        dict: Simulation results containing:
          - avg_re: Long-term average reconstruction error.
          - freq1: Long-term average transmission frequency of Sensor 1.
          - freq2: Long-term average transmission frequency of Sensor 2.
          - single1: Frequency of Sensor 1 transmitting alone (if count_actions=True).
          - single2: Frequency of Sensor 2 transmitting alone (if count_actions=True).
          - both: Frequency of both sensors transmitting simultaneously (if count_actions=True).
          - none: Frequency of no sensors transmitting (if count_actions=True).
    """
    S = cmdp.num_states  # Total number of system states
    A = cmdp.num_actions  # Total number of joint actions
    state = 0  # Initial state index (starts from state 0)
    total_re = 0.0  # Accumulated reconstruction error over effective steps
    total_f1 = 0  # Accumulated transmission count for Sensor 1
    total_f2 = 0  # Accumulated transmission count for Sensor 2
    cnt = 0  # Counter for effective steps

    # Dictionary to track transmission mode statistics
    stat = {
        "single1": 0,  # Sensor 1 transmits alone
        "single2": 0,  # Sensor 2 transmits alone
        "both": 0,    # Both sensors transmit simultaneously
        "none": 0,    # No sensors transmit
    }

    # Combined burn-in and effective simulation loop
    for t in range(num_steps + burn_in):
        # Select action based on the policy's probability distribution for the current state
        probs = policy[state]
        a_idx = np.random.choice(A, p=probs)
        a1, a2 = cmdp.actions[a_idx]  # Get the actual joint action (a1: Sensor 1's action, a2: Sensor 2's action)

        # Sample next state according to the transition kernel P[s, a, :]
        next_state = np.random.choice(S, p=cmdp.P[state, a_idx])

        # Calculate one-step reconstruction error (RE) for the current state-action pair
        re = cmdp.c_sa[state, a_idx]

        # Accumulate statistics only after burn-in phase
        if t >= burn_in:
            total_re += re
            total_f1 += (1 if a1 != 0 else 0)  # 1 if Sensor 1 transmits, 0 otherwise
            total_f2 += (1 if a2 != 0 else 0)  # 1 if Sensor 2 transmits, 0 otherwise
            cnt += 1

            # Update transmission mode statistics if enabled
            if count_actions:
                if a1 != 0 and a2 == 0:
                    stat["single1"] += 1
                elif a1 == 0 and a2 != 0:
                    stat["single2"] += 1
                elif a1 != 0 and a2 != 0:
                    stat["both"] += 1
                else:
                    stat["none"] += 1

        # Update current state to the next state
        state = next_state

    # Normalize transmission mode statistics to frequencies (0~1)
    for k in stat:
        stat[k] /= cnt

    return {
        "avg_re": total_re / cnt,
        "freq1": total_f1 / cnt,
        "freq2": total_f2 / cnt,
        "single1": stat["single1"],
        "single2": stat["single2"],
        "both": stat["both"],
        "none": stat["none"],
    }


# ============================================================
# Wrapper MC Evaluation Interface for LP / RVI / FPS
# ============================================================
def mc_eval_policy(cmdp, policy, steps, burn_in):
    """
    Wrapper function to evaluate LP, RVI, or FPS policies using Monte-Carlo simulation.
    Simplifies interface by directly calling simulate_policy_mc with count_actions=True.

    Parameters:
        cmdp (TwoSourceMPRCMDP): Instance of the two-source MPR-CMDP model.
        policy (np.ndarray): Randomized policy matrix (output of LP/RVI/FPS algorithms).
        steps (int): Number of effective simulation steps.
        burn_in (int): Burn-in steps to reach steady state.

    Returns:
        dict: Simulation results (same format as simulate_policy_mc).
    """
    res = simulate_policy_mc(cmdp, policy, steps, burn_in, True)
    return res


# ============================================================
# Main Evaluation Function: RE vs Fmax
# ============================================================
def evaluate_re_vs_fmax(p_fixed=0.5, V_dpp=100, steps=20000, burn_in=2000):
    """
    Evaluate the performance of four algorithms (LP, RVI, FPS, DPP) by varying the maximum
    allowable transmission frequency (Fmax). The single-sensor transmission success probability (p) is fixed.
    Reconstruction Error (RE) is used as the performance metric (delta matrices set for RE calculation).

    Parameters:
        p_fixed (float): Fixed success probability for single-sensor transmissions (p1_only = p2_only = p_fixed).
                         Simultaneous transmission success probability is p_fixed / 2.
        V_dpp (int): Weight factor for RE minimization in the DPP algorithm.
        steps (int): Number of effective simulation steps per Fmax value.
        burn_in (int): Burn-in steps per Fmax value.

    Returns:
        list: A list of dictionaries, where each dictionary contains results for a specific Fmax value.
    """
    results = []
    f_list = np.arange(0.1, 1.01, 0.1)  # Range of Fmax values (0.1 to 1.0, step 0.1)
    print(f"\n====== Running: RE vs Fmax (p={p_fixed}) ======\n")

    # Iterate over each Fmax value
    for fmax in f_list:
        print(f">>> Fmax = {fmax:.1f}")
        # Initialize the two-source MPR-CMDP model with current Fmax constraint and RE-focused delta matrices
        cmdp = TwoSourceMPRCMDP(
            p1_only=p_fixed,       # Success prob of Sensor 1 transmitting alone
            p2_only=p_fixed,       # Success prob of Sensor 2 transmitting alone
            p1_both=p_fixed / 2,    # Success prob of Sensor 1 when both transmit
            p2_both=p_fixed / 2,    # Success prob of Sensor 2 when both transmit
            Fmax1=fmax,            # Max transmission frequency for Sensor 1
            Fmax2=fmax,            # Max transmission frequency for Sensor 2
            delta1=np.array([[0, 1], [1, 0]]),  # RE cost matrix for Source 1 (|i-j|)
            delta2=np.array([[0, 1], [1, 0]])   # RE cost matrix for Source 2 (|i-j|)
        )

        row = {"Fmax": fmax}  # Store current Fmax value

        # ---------------- Evaluate LP Algorithm ----------------
        try:
            # Solve CMDP via occupation-measure LP and get optimal policy
            X, pi_lp, _, C_lp, _, _ = solve_cmdp_via_lp(cmdp)
            # Evaluate LP policy using Monte-Carlo simulation
            mc_lp = mc_eval_policy(cmdp, pi_lp, steps, burn_in)
            row["LP"] = mc_lp
            print(f"  LP   RE={mc_lp['avg_re']:.4f}")
        except Exception as e:
            print("  LP FAILED:", e)
            row["LP"] = None  # Mark as failed if exception occurs

        # ---------------- Evaluate RVI Algorithm ----------------
        try:
            # Solve CMDP via dual bisection search RVI and get mixed optimal policy
            rvi_res = cmdp.double_bisection_rvi()
            pi_rvi = rvi_res["mix_policy"]
            # Evaluate RVI policy using Monte-Carlo simulation
            mc_rvi = mc_eval_policy(cmdp, pi_rvi, steps, burn_in)
            row["RVI"] = mc_rvi
            print(f"  RVI  RE={mc_rvi['avg_re']:.4f}")
        except Exception as e:
            print("  RVI FAILED:", e)
            row["RVI"] = None  # Mark as failed if exception occurs

        # ---------------- Evaluate FPS Algorithm ----------------
        try:
            # Build fixed-probability sampling policy
            pi_fix = build_fixed_policy(cmdp)
            # Evaluate FPS policy using Monte-Carlo simulation
            mc_fix = mc_eval_policy(cmdp, pi_fix, steps, burn_in)
            row["FPS"] = mc_fix
            print(f"  FPS  RE={mc_fix['avg_re']:.4f}")
        except Exception as e:
            print("  FPS FAILED:", e)
            row["FPS"] = None  # Mark as failed if exception occurs

        # ---------------- Evaluate DPP Algorithm ----------------
        # Initialize DPP algorithm with specified weight factor
        dpp = LowComplexityDPP(cmdp, V1=V_dpp, V2=V_dpp)
        # Run DPP simulation (integrated simulation in LowComplexityDPP class)
        avg_re, avg_freq = dpp.run_simulation(
            num_steps=steps,
            burn_in=burn_in,
            count_actions=True
        )
        # Store DPP results in consistent format
        row["DPP"] = {
            "avg_re": avg_re,
            "freq1": avg_freq[0],
            "freq2": avg_freq[1],
            "single1": dpp.stats["single1"],
            "single2": dpp.stats["single2"],
            "both": dpp.stats["both"],
            "none": dpp.stats["none"],
        }
        print(f"  DPP  RE={avg_re:.4f}")

        # Add results for current Fmax to the global results list
        results.append(row)

    return results


# ============================================================
# New: Save Results Split by Algorithm (MATLAB-Ready Indexing)
# ============================================================
def save_results_for_matlab(results, mat_filename="re_vs_fmax.mat", csv_filename="re_vs_fmax.csv"):
    """
    Save simulation results in a MATLAB-friendly format, where each algorithm's metrics are stored as
    independent 1D arrays (dimension = number of Fmax values = 10).

    Example: lp_re = [value at Fmax=0.1, value at Fmax=0.2, ..., value at Fmax=1.0]

    Parameters:
        results (list): Output of evaluate_re_vs_fmax, containing results for each Fmax value.
        mat_filename (str): Path to save the MAT file (default: "re_vs_fmax.mat").
        csv_filename (str): Path to save the backup CSV file (default: "re_vs_fmax.csv").
    """
    # 1. Extract Fmax values (0.1 to 1.0, 10 values total)
    Fmax = np.array([row["Fmax"] for row in results], dtype=np.float64)

    # 2. Initialize metric arrays for each algorithm (filled with NaN for missing values)
    alg_list = ["LP", "RVI", "FPS", "DPP"]
    metrics = ["RE", "F1", "F2", "single1", "single2", "both", "none"]

    # Create dictionary to store MATLAB-compatible data
    mat_data = {"Fmax": Fmax}  # First add Fmax as a reference
    for alg in alg_list:
        for metric in metrics:
            # Key format: algorithm_metric (e.g., lp_re, rvi_f1)
            mat_data[f"{alg.lower()}_{metric.lower()}"] = np.full_like(Fmax, np.nan, dtype=np.float64)

    # 3. Populate data (fill metrics for each algorithm and Fmax)
    for f_idx, row in enumerate(results):
        for alg in alg_list:
            result_alg = row[alg]
            if result_alg is None:
                continue  # Skip if algorithm failed for current Fmax (keep NaN)
            # Fill all metrics for the current algorithm and Fmax
            mat_data[f"{alg.lower()}_re"][f_idx] = result_alg["avg_re"]
            mat_data[f"{alg.lower()}_f1"][f_idx] = result_alg["freq1"]
            mat_data[f"{alg.lower()}_f2"][f_idx] = result_alg["freq2"]
            mat_data[f"{alg.lower()}_single1"][f_idx] = result_alg["single1"]
            mat_data[f"{alg.lower()}_single2"][f_idx] = result_alg["single2"]
            mat_data[f"{alg.lower()}_both"][f_idx] = result_alg["both"]
            mat_data[f"{alg.lower()}_none"][f_idx] = result_alg["none"]

    # 4. Save as MAT file (directly loadable in MATLAB)
    sio.savemat(mat_filename, mat_data)
    print(f"\nData split by algorithm saved in MATLAB format: {mat_filename}")

    # 5. Backup: Save as CSV file (split by algorithm + metric columns)
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        # Construct CSV headers: Fmax, lp_re, lp_f1, ..., dpp_none
        csv_headers = ["Fmax"]
        for alg in alg_list:
            for metric in metrics:
                csv_headers.append(f"{alg.lower()}_{metric.lower()}")
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        # Write each row (one row per Fmax value)
        for f_idx in range(len(Fmax)):
            csv_row = {"Fmax": Fmax[f_idx]}
            for alg in alg_list:
                for metric in metrics:
                    csv_row[f"{alg.lower()}_{metric.lower()}"] = mat_data[f"{alg.lower()}_{metric.lower()}"][f_idx]
            writer.writerow(csv_row)
    print(f"Data split by algorithm saved in CSV format: {csv_filename}")


# ============================================================
# Print Results Table
# ============================================================
def print_results_table(results):
    """
    Print a formatted table of simulation results, including reconstruction error (RE), transmission frequencies,
    and transmission mode statistics for all algorithms and Fmax values.
    """
    print("\n" + "=" * 150)
    print(" FINAL RESULT TABLE (includes single/both action frequencies)")
    print("=" * 150)
    # Table header
    print(f"{'Fmax':<6}{'Alg':<6}{'RE':<10}{'F1':<8}{'F2':<8}"
          f"{'single1':<10}{'single2':<10}{'both':<10}{'none':<10}")
    print("-" * 150)

    # Print results for each Fmax and algorithm
    for row in results:
        F = row["Fmax"]
        for alg in ["LP", "RVI", "FPS", "DPP"]:
            result_alg = row[alg]
            if result_alg is None:
                print(f"{F:<6.1f}{alg:<6}{'FAIL':<10}")
                continue
            # Print formatted metrics (4 decimal places for RE, 3 for others)
            print(f"{F:<6.1f}{alg:<6}"
                  f"{result_alg['avg_re']:<10.4f}"
                  f"{result_alg['freq1']:<8.3f}{result_alg['freq2']:<8.3f}"
                  f"{result_alg['single1']:<10.3f}{result_alg['single2']:<10.3f}"
                  f"{result_alg['both']:<10.3f}{result_alg['none']:<10.3f}")
    print("=" * 150)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Run the main evaluation (RE vs Fmax)
    results = evaluate_re_vs_fmax(
        p_fixed=0.8,    # Fixed single-sensor transmission success probability
        V_dpp=100,      # Weight factor for DPP algorithm
        steps=200000,   # Effective simulation steps per Fmax
        burn_in=20000   # Burn-in steps per Fmax
    )

    # Print formatted results table
    print_results_table(results)

    # Save results split by algorithm (MATLAB-ready and CSV backup)
    save_results_for_matlab(results, "result/re_vs_fmax.mat")