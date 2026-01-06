import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from CAEvsP_compare import *   # Contains TwoSourceMPRCMDP, solve_cmdp_via_lp, build_fixed_policy, etc.
from DPP import *              # Contains LowComplexityDPP


# ============================================================
# 1. Monte-Carlo Simulation Function (Independent of simulate_transmission)
# ============================================================
def simulate_policy_mc(cmdp, policy, num_steps=50000, burn_in=2000, count_actions=True):
    """
    Monte-Carlo simulation for a given stationary randomized policy.
    Compatible with the TwoSourceMPRCMDP class even if it does NOT have a simulate_transmission method.

    Parameters:
        cmdp (TwoSourceMPRCMDP): Instance of the two-source MPR-CMDP model.
        policy (np.ndarray): Randomized policy matrix with shape (S, A), where each row (corresponding to a state)
                             is a probability distribution over the 9 joint actions.
        num_steps (int): Number of effective steps for statistics collection.
        burn_in (int): Burn-in steps (not included in final statistics) to reach steady state.
        count_actions (bool): Whether to count the frequency of transmission modes (single1/single2/both/none).

    Returns:
        dict: Simulation results containing:
          - avg_cae: Long-term average Cost of Actuation Error.
          - freq1: Long-term average transmission frequency of Sensor 1.
          - freq2: Long-term average transmission frequency of Sensor 2.
          - single1: Frequency of Sensor 1 transmitting alone (if count_actions=True).
          - single2: Frequency of Sensor 2 transmitting alone (if count_actions=True).
          - both: Frequency of both sensors transmitting simultaneously (if count_actions=True).
          - none: Frequency of no sensors transmitting (if count_actions=True).
    """

    S = cmdp.num_states  # Total number of system states
    A = cmdp.num_actions  # Total number of joint actions (9 for 2 sensors × 3 actions each)
    actions = cmdp.actions     # List of joint actions, each element is a tuple (a1, a2)
    P = cmdp.P                 # Transition kernel with shape (S, A, S), P[s,a,s'] is transition probability
    c_sa = cmdp.c_sa           # One-step expected CAE matrix with shape (S, A)

    # Initialize system state (start from state 0 by default)
    state = 0
    total_cae = 0.0  # Accumulated CAE over effective steps
    total_f1 = 0     # Accumulated transmission count for Sensor 1
    total_f2 = 0     # Accumulated transmission count for Sensor 2
    cnt = 0          # Counter for effective steps

    # Dictionary to store transmission mode statistics
    stats = {"single1": 0, "single2": 0, "both": 0, "none": 0}

    # ---------------- Burn-in Phase (Reach Steady State) ----------------
    for _ in range(burn_in):
        # Randomly select action based on the policy's probability distribution for current state
        a_idx = np.random.choice(A, p=policy[state])
        # Transition to next state based on the transition kernel
        state = np.random.choice(S, p=P[state, a_idx])

    # ---------------- Formal Simulation Phase ----------------
    for _ in range(num_steps):
        # Select action according to the policy
        a_idx = np.random.choice(A, p=policy[state])
        a1, a2 = actions[a_idx]  # Get the actual joint action (a1: action of Sensor 1, a2: action of Sensor 2)

        # Transition to next state
        next_state = np.random.choice(S, p=P[state, a_idx])

        # Accumulate CAE for current state-action pair
        cae = c_sa[state, a_idx]
        total_cae += cae

        # Check if each sensor transmits (1 = transmit, 0 = silent)
        f1 = 1 if a1 != 0 else 0
        f2 = 1 if a2 != 0 else 0
        total_f1 += f1
        total_f2 += f2

        # Count transmission modes if enabled
        if count_actions:
            if f1 and not f2:
                stats["single1"] += 1
            elif f2 and not f1:
                stats["single2"] += 1
            elif f1 and f2:
                stats["both"] += 1
            else:
                stats["none"] += 1

        # Update step counter and current state
        cnt += 1
        state = next_state

    # ---------------- Compile Results ----------------
    result = {
        "avg_cae": total_cae / cnt,  # Average CAE over effective steps
        "freq1": total_f1 / cnt,     # Average transmission frequency of Sensor 1
        "freq2": total_f2 / cnt,     # Average transmission frequency of Sensor 2
    }

    # Add transmission mode statistics if enabled (normalized to [0,1])
    if count_actions:
        for k in stats:
            stats[k] /= cnt
        result.update(stats)

    return result


# ============================================================
# 2. Main Evaluation Function: CAE vs p_both (LP / RVI / DPP / FPS)
# ============================================================
def evaluate_cae_vs_p_both(
    Fmax_fixed=0.6,
    p_only_fixed=0.9,
    V_dpp=100,
    steps=100000,
    burn_in=20000,
):
    """
    Evaluate the performance of four algorithms (LP, RVI, DPP, FPS) by varying the simultaneous transmission
    success probability (p_both). The evaluation focuses on long-term average CAE, transmission frequencies,
    and transmission mode distributions.

    Parameters:
        Fmax_fixed (float): Maximum allowable long-term transmission frequency for both sensors.
        p_only_fixed (float): Success probability of transmission when only one sensor transmits.
        V_dpp (int): Weight factor for CAE minimization in the DPP algorithm.
        steps (int): Effective simulation steps per p_both value.
        burn_in (int): Burn-in steps per p_both value.

    Returns:
        list: A list of dictionaries, where each dictionary contains results for a specific p_both value.
    """
    results = []
    # Range of simultaneous transmission success probabilities to test (0.1 to 0.9, step 0.1)
    p_both_list = np.arange(0.1, 0.91, 0.1)

    print(f"\n=== Test Configuration: Fmax={Fmax_fixed}, p_only={p_only_fixed}, V_dpp={V_dpp} ===\n")

    # Iterate over each p_both value
    for p_both in p_both_list:
        print(f">>> Starting evaluation for p_both = {p_both:.1f}")
        row = {"p_both": float(p_both)}  # Store current p_both value

        # ----------------------------------------------------
        # Build the Two-Source MPR-CMDP Model
        # ----------------------------------------------------
        cmdp = TwoSourceMPRCMDP(
            p1_only=p_only_fixed,    # Success prob of Sensor 1 transmitting alone
            p2_only=p_only_fixed,    # Success prob of Sensor 2 transmitting alone
            p1_both=p_both,          # Success prob of Sensor 1 when both transmit
            p2_both=p_both,          # Success prob of Sensor 2 when both transmit
            Fmax1=Fmax_fixed,        # Max transmission frequency for Sensor 1
            Fmax2=Fmax_fixed,        # Max transmission frequency for Sensor 2
        )

        # ----------------------------------------------------
        # 1) LP Optimal Algorithm
        # ----------------------------------------------------
        try:
            # Solve CMDP via occupation-measure LP and get optimal randomized policy
            _, pi_lp, _, _, _, _ = solve_cmdp_via_lp(cmdp)
            # Run Monte-Carlo simulation for the LP policy
            mc_lp = simulate_policy_mc(cmdp, pi_lp, steps, burn_in, count_actions=True)
            row["LP"] = mc_lp  # Store simulation results

            # Print LP results
            print(
                f"  [LP ] CAE={mc_lp['avg_cae']:.4f} | "
                f"F1={mc_lp['freq1']:.3f}, F2={mc_lp['freq2']:.3f} | "
                f"single1={mc_lp['single1']:.3f}, single2={mc_lp['single2']:.3f}, "
                f"both={mc_lp['both']:.3f}, none={mc_lp['none']:.3f}"
            )
        except Exception as e:
            print(f"  [LP ] Failed: {e}")
            row["LP"] = None  # Mark as failed if exception occurs

        # ----------------------------------------------------
        # 2) RVI Algorithm (Dual Bisection Search)
        # ----------------------------------------------------
        try:
            # Solve CMDP via dual bisection search RVI and get mixed optimal policy
            rvi_res = cmdp.double_bisection_rvi()
            pi_rvi = rvi_res["mix_policy"]  # Randomized policy from RVI

            # Run Monte-Carlo simulation for the RVI policy
            mc_rvi = simulate_policy_mc(cmdp, pi_rvi, steps, burn_in, count_actions=True)
            row["RVI"] = mc_rvi  # Store simulation results

            # Print RVI results
            print(
                f"  [RVI] CAE={mc_rvi['avg_cae']:.4f} | "
                f"F1={mc_rvi['freq1']:.3f}, F2={mc_rvi['freq2']:.3f} | "
                f"single1={mc_rvi['single1']:.3f}, single2={mc_rvi['single2']:.3f}, "
                f"both={mc_rvi['both']:.3f}, none={mc_rvi['none']:.3f}"
            )
        except Exception as e:
            print(f"  [RVI] Failed: {e}")
            row["RVI"] = None  # Mark as failed if exception occurs

        # ----------------------------------------------------
        # 3) FPS Algorithm
        # ----------------------------------------------------
        try:
            # Build fixed-probability sampling policy (implemented in CAEvsP_compare)
            pi_fps = build_fixed_policy(cmdp)
            # Run Monte-Carlo simulation for the FPS policy
            mc_fps = simulate_policy_mc(cmdp, pi_fps, steps, burn_in, count_actions=True)
            row["FPS"] = mc_fps  # Store simulation results

            # Print FPS results
            print(
                f"  [FPS] CAE={mc_fps['avg_cae']:.4f} | "
                f"F1={mc_fps['freq1']:.3f}, F2={mc_fps['freq2']:.3f} | "
                f"single1={mc_fps['single1']:.3f}, single2={mc_fps['single2']:.3f}, "
                f"both={mc_fps['both']:.3f}, none={mc_fps['none']:.3f}"
            )
        except Exception as e:
            print(f"  [FPS] Failed: {e}")
            row["FPS"] = None  # Mark as failed if exception occurs

        # ----------------------------------------------------
        # 4) DPP Algorithm
        # ----------------------------------------------------
        try:
            # Initialize DPP algorithm with specified weight factor
            dpp = LowComplexityDPP(cmdp, V1=V_dpp, V2=V_dpp)
            # Run DPP simulation (integrated simulation in LowComplexityDPP class)
            avg_cae, (F1, F2) = dpp.run_simulation(
                num_steps=steps,
                burn_in=burn_in,
                count_actions=True
            )

            # Store DPP results (consistent with other algorithms' output format)
            row["DPP"] = {
                "avg_cae": avg_cae,
                "freq1": F1,
                "freq2": F2,
                "single1": dpp.stats["single1"],
                "single2": dpp.stats["single2"],
                "both": dpp.stats["both"],
                "none": dpp.stats["none"],
            }

            # Print DPP results
            print(
                f"  [DPP] CAE={avg_cae:.4f} | "
                f"F1={F1:.3f}, F2={F2:.3f} | "
                f"single1={dpp.stats['single1']:.3f}, single2={dpp.stats['single2']:.3f}, "
                f"both={dpp.stats['both']:.3f}, none={dpp.stats['none']:.3f}"
            )
        except Exception as e:
            print(f"  [DPP] Failed: {e}")
            row["DPP"] = None  # Mark as failed if exception occurs

        print("-" * 100)
        results.append(row)  # Add current p_both results to the global list

    return results


# ============================================================
# 3. Result Saving + Summary Printing
# ============================================================
def save(results, save_path="cae_vs_p_both.mat"):
    """
    Save simulation results to a MAT file for further analysis, and print a summary table to the console.

    Parameters:
        results (list): List of dictionaries containing results for each p_both value (output of evaluate_cae_vs_p_both).
        save_path (str): Path to save the MAT file (default: "cae_vs_p_both.mat").
    """
    # Extract p_both values from results
    p_both = np.array([r["p_both"] for r in results])

    # Helper function to extract specific metrics from algorithm results
    def extract(algo, key):
        arr = []
        for r in results:
            if r.get(algo) is None:
                arr.append(np.nan)  # Use NaN for failed runs
            else:
                arr.append(r[algo][key])
        return np.array(arr, dtype=float)

    # ------------- Extract Metrics for All Algorithms -------------
    algorithms = ["LP", "RVI", "FPS", "DPP"]
    metrics = ["avg_cae", "freq1", "freq2", "single1", "single2", "both", "none"]

    # Organize data into a dictionary (MAT file compatible)
    data = {}
    for algo in algorithms:
        for m in metrics:
            key = f"{algo.lower()}_{m}"  # Key format: algo_metric (e.g., lp_avg_cae)
            data[key] = extract(algo, m)

    # ---------------- Save to MAT File ----------------
    mat_dict = {"p_both": p_both}
    mat_dict.update(data)  # Combine p_both and algorithm metrics
    sio.savemat(save_path, mat_dict)
    print(f"\nMAT file saved successfully: {save_path}")

    # ---------------- Print Summary Table to Console ----------------
    print("\n" + "=" * 160)
    print("  Performance Summary Table: CAE / F1 / F2 / single1 / single2 / both / none for Each Algorithm")
    print("=" * 160)
    # Define table header
    header = (
        f"{'p_both':<7} | "
        f"{'Algo':<5} | "
        f"{'CAE':<10} {'F1':<10} {'F2':<10} "
        f"{'single1':<10} {'single2':<10} {'both':<10} {'none':<10}"
    )
    print(header)
    print("-" * 160)

    # Print results for each p_both and algorithm
    for i, pb in enumerate(p_both):
        for algo in algorithms:
            prefix = algo.lower()
            # Extract metrics for current algorithm and p_both
            cae = data[f"{prefix}_avg_cae"][i]
            f1 = data[f"{prefix}_freq1"][i]
            f2 = data[f"{prefix}_freq2"][i]
            s1 = data[f"{prefix}_single1"][i]
            s2 = data[f"{prefix}_single2"][i]
            bt = data[f"{prefix}_both"][i]
            ne = data[f"{prefix}_none"][i]

            # Handle failed runs (NaN values)
            if np.isnan(cae):
                line = f"{pb:<7.1f} | {algo:<5} | " + "N/A".ljust(10) * 7
            else:
                # Format numbers to 4 decimal places for consistency
                line = (
                    f"{pb:<7.1f} | {algo:<5} | "
                    f"{cae:<10.4f}{f1:<10.4f}{f2:<10.4f}"
                    f"{s1:<10.4f}{s2:<10.4f}{bt:<10.4f}{ne:<10.4f}"
                )
            print(line)

    print("=" * 160)


# ============================================================
# 4. Main Entry Point
# ============================================================
if __name__ == "__main__":
    # Run the main evaluation (CAE vs p_both)
    results = evaluate_cae_vs_p_both(
        Fmax_fixed=0.6,    # Maximum transmission frequency constraint
        p_only_fixed=0.9,  # Success probability for single-sensor transmission
        V_dpp=100,         # DPP weight for CAE minimization
        steps=120000,      # Effective simulation steps
        burn_in=20000,     # Burn-in steps
    )

    # Save results to MAT file and print summary
    save(results, save_path="result/cae_vs_p_both.mat")