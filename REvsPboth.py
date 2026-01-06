import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from REvsP import *  # Contains TwoSourceMPRCMDP, solve_cmdp_via_lp, build_fixed_policy, etc.
from DPP import *  # Contains LowComplexityDPP


# ============================================================
# 1. Monte-Carlo Simulation Function
# ============================================================
def simulate_policy_mc(cmdp, policy, num_steps=50000, burn_in=2000, count_actions=True):
    """
    Monte-Carlo simulation for a given policy.
    Compatible with TwoSourceMPRCMDP even if it has NO simulate_transmission method.

    Parameters:
        cmdp: TwoSourceMPRCMDP object
        policy: Stochastic policy with shape (S, A), where each row (per state)
                represents the probability distribution over 9 joint actions
        num_steps: Number of steps for formal statistics collection
        burn_in: Number of burn-in steps (not included in statistics)
        count_actions: Whether to count frequencies of single1 / single2 / both / none actions

    Returns:
        dict:
          - avg_re: Average RE (Radio Exposure) cost
          - freq1: Frequency of source 1 transmission
          - freq2: Frequency of source 2 transmission
          - single1, single2, both, none: Action type frequencies (if count_actions=True)
    """

    S = cmdp.num_states
    A = cmdp.num_actions
    actions = cmdp.actions  # list of (a1,a2) joint actions
    P = cmdp.P  # Transition probability matrix P[s,a,s']
    c_sa = cmdp.c_sa  # RE cost array (state-action dependent)

    state = 0
    total_re = 0.0
    total_f1 = 0
    total_f2 = 0
    cnt = 0

    stats = {"single1": 0, "single2": 0, "both": 0, "none": 0}

    # ---------------- Burn-in Phase ----------------
    for _ in range(burn_in):
        a_idx = np.random.choice(A, p=policy[state])
        state = np.random.choice(S, p=P[state, a_idx])

    # ---------------- Formal Simulation ----------------
    for _ in range(num_steps):
        a_idx = np.random.choice(A, p=policy[state])
        a1, a2 = actions[a_idx]

        next_state = np.random.choice(S, p=P[state, a_idx])

        # Accumulate RE cost
        re = c_sa[state, a_idx]
        total_re += re

        # Count transmission frequencies
        f1 = 1 if a1 != 0 else 0
        f2 = 1 if a2 != 0 else 0
        total_f1 += f1
        total_f2 += f2

        # Count action type statistics
        if count_actions:
            if f1 and not f2:
                stats["single1"] += 1
            elif f2 and not f1:
                stats["single2"] += 1
            elif f1 and f2:
                stats["both"] += 1
            else:
                stats["none"] += 1

        cnt += 1
        state = next_state

    # ---------------- Result Calculation ----------------
    result = {
        "avg_re": total_re / cnt,
        "freq1": total_f1 / cnt,
        "freq2": total_f2 / cnt,
    }

    # Normalize action statistics to frequencies
    if count_actions:
        for k in stats:
            stats[k] /= cnt
        result.update(stats)

    return result


# ============================================================
# 2. RE vs p_both: Main Evaluation Function for LP/RVI/DPP/FPS
# ============================================================
def evaluate_re_vs_p_both(
        Fmax_fixed=0.6,
        p_only_fixed=0.9,
        V_dpp=100,
        steps=100000,
        burn_in=20000,
):
    """
    Evaluate and compare different algorithms (LP/RVI/DPP/FPS)
    on RE (Radio Exposure) vs p_both (transmission probability under both sources active)

    Parameters:
        Fmax_fixed: Maximum allowable transmission frequency constraint
        p_only_fixed: Transmission probability when only one source is active
        V_dpp: DPP (Dynamic Policy Programming) parameter V
        steps: Number of Monte-Carlo simulation steps
        burn_in: Burn-in steps for simulation

    Returns:
        List of dictionaries containing evaluation results for each p_both value
    """
    results = []
    p_both_list = np.arange(0.1, 0.91, 0.1)  # p_both values from 0.1 to 0.9 with 0.1 step

    print(f"\n=== Test Configuration: Fmax={Fmax_fixed}, p_only={p_only_fixed}, V_dpp={V_dpp} ===\n")

    for p_both in p_both_list:
        print(f">>> Starting evaluation for p_both = {p_both:.1f}")
        row = {"p_both": float(p_both)}

        # ----------------------------------------------------
        # Build CMDP Model
        # ----------------------------------------------------
        cmdp = TwoSourceMPRCMDP(
            p1_only=p_only_fixed,
            p2_only=p_only_fixed,
            p1_both=p_both,
            p2_both=p_both,
            Fmax1=Fmax_fixed,
            Fmax2=Fmax_fixed,
            delta1=np.array([[0, 1],
                             [1, 0]]),
            delta2=np.array([[0, 1],
                             [1, 0]])
        )

        # ----------------------------------------------------
        # 1) LP OPTIMAL (Linear Programming)
        # ----------------------------------------------------
        try:
            _, pi_lp, _, _, _, _ = solve_cmdp_via_lp(cmdp)

            mc_lp = simulate_policy_mc(cmdp, pi_lp, steps, burn_in, count_actions=True)
            row["LP"] = mc_lp

            print(
                f"  [LP ] RE={mc_lp['avg_re']:.4f} | "
                f"F1={mc_lp['freq1']:.3f}, F2={mc_lp['freq2']:.3f} | "
                f"single1={mc_lp['single1']:.3f}, single2={mc_lp['single2']:.3f}, "
                f"both={mc_lp['both']:.3f}, none={mc_lp['none']:.3f}"
            )
        except Exception as e:
            print("  [LP ] Failed:", e)
            row["LP"] = None

        # ----------------------------------------------------
        # 2) RVI (Relative Value Iteration with double-bisection)
        # ----------------------------------------------------
        try:
            cmdp.delta1 = np.array([[0, 1],
                                    [1, 0]])
            cmdp.delta2 = np.array([[0, 1],
                                    [1, 0]])
            rvi_res = cmdp.double_bisection_rvi()
            pi_rvi = rvi_res["mix_policy"]

            mc_rvi = simulate_policy_mc(cmdp, pi_rvi, steps, burn_in, count_actions=True)
            row["RVI"] = mc_rvi

            print(
                f"  [RVI] RE={mc_rvi['avg_re']:.4f} | "
                f"F1={mc_rvi['freq1']:.3f}, F2={mc_rvi['freq2']:.3f} | "
                f"single1={mc_rvi['single1']:.3f}, single2={mc_rvi['single2']:.3f}, "
                f"both={mc_rvi['both']:.3f}, none={mc_rvi['none']:.3f}"
            )
        except Exception as e:
            print("  [RVI] Failed:", e)
            row["RVI"] = None

        # ----------------------------------------------------
        # 3) FPS (Fixed Probability Scheduling)
        # ----------------------------------------------------
        try:
            cmdp.delta1 = np.array([[0, 1],
                                    [1, 0]])
            cmdp.delta2 = np.array([[0, 1],
                                    [1, 0]])
            pi_fps = build_fixed_policy(cmdp)  # Fixed probability policy implemented in REvsP_compare
            mc_fps = simulate_policy_mc(cmdp, pi_fps, steps, burn_in, count_actions=True)
            row["FPS"] = mc_fps

            print(
                f"  [FPS] RE={mc_fps['avg_re']:.4f} | "
                f"F1={mc_fps['freq1']:.3f}, F2={mc_fps['freq2']:.3f} | "
                f"single1={mc_fps['single1']:.3f}, single2={mc_fps['single2']:.3f}, "
                f"both={mc_fps['both']:.3f}, none={mc_fps['none']:.3f}"
            )
        except Exception as e:
            print("  [FPS] Failed:", e)
            row["FPS"] = None

        # ----------------------------------------------------
        # 4) DPP (Low Complexity Dynamic Policy Programming)
        # ----------------------------------------------------
        try:
            dpp = LowComplexityDPP(cmdp, V1=V_dpp, V2=V_dpp)
            avg_re, (F1, F2) = dpp.run_simulation(
                num_steps=steps,
                burn_in=burn_in,
                count_actions=True
            )

            row["DPP"] = {
                "avg_re": avg_re,
                "freq1": F1,
                "freq2": F2,
                "single1": dpp.stats["single1"],
                "single2": dpp.stats["single2"],
                "both": dpp.stats["both"],
                "none": dpp.stats["none"],
            }

            print(
                f"  [DPP] RE={avg_re:.4f} | "
                f"F1={F1:.3f}, F2={F2:.3f} | "
                f"single1={dpp.stats['single1']:.3f}, single2={dpp.stats['single2']:.3f}, "
                f"both={dpp.stats['both']:.3f}, none={dpp.stats['none']:.3f}"
            )
        except Exception as e:
            print("  [DPP] Failed:", e)
            row["DPP"] = None

        print("-" * 100)
        results.append(row)

    return results


# ============================================================
# 3. Visualization + Data Saving + Summary Printing
# ============================================================
def visualize_and_save(results, save_path="re_vs_p_both.mat"):
    """
    Process evaluation results: extract metrics, save to MAT file,
    and print comprehensive summary table.

    Parameters:
        results: Evaluation results from evaluate_re_vs_p_both
        save_path: Path to save MAT format data file
    """
    # Extract p_both values
    p_both = np.array([r["p_both"] for r in results])

    # Helper function to extract specific metric for an algorithm
    def extract(algo, key):
        arr = []
        for r in results:
            if r.get(algo) is None:
                arr.append(np.nan)
            else:
                arr.append(r[algo][key])
        return np.array(arr, dtype=float)

    # ------------- Extract Metrics for All Algorithms -------------
    algorithms = ["LP", "RVI", "FPS", "DPP"]
    metrics = ["avg_re", "freq1", "freq2", "single1", "single2", "both", "none"]

    data = {}
    for algo in algorithms:
        for m in metrics:
            key = f"{algo.lower()}_{m}"
            data[key] = extract(algo, m)

    # ---------------- Save to MAT File ----------------
    mat_dict = {"p_both": p_both}
    mat_dict.update(data)
    sio.savemat(save_path, mat_dict)
    print(f"\nMAT file saved to: {save_path}")

    # ---------------- Print Summary Table to Console ----------------
    print("\n" + "=" * 160)
    print("  Performance Summary Table: RE / F1 / F2 / single1 / single2 / both / none for each algorithm")
    print("=" * 160)
    header = (
        f"{'p_both':<7} | "
        f"{'Algo':<5} | "
        f"{'RE':<10} {'F1':<10} {'F2':<10} "
        f"{'single1':<10} {'single2':<10} {'both':<10} {'none':<10}"
    )
    print(header)
    print("-" * 160)

    # Print row by row for each p_both and algorithm
    for i, pb in enumerate(p_both):
        for algo in algorithms:
            prefix = algo.lower()
            re = data[f"{prefix}_avg_re"][i]
            f1 = data[f"{prefix}_freq1"][i]
            f2 = data[f"{prefix}_freq2"][i]
            s1 = data[f"{prefix}_single1"][i]
            s2 = data[f"{prefix}_single2"][i]
            bt = data[f"{prefix}_both"][i]
            ne = data[f"{prefix}_none"][i]

            if np.isnan(re):
                # Algorithm failed for this p_both value
                line = f"{pb:<7.1f} | {algo:<5} | " + "N/A".ljust(10) * 7
            else:
                line = (
                    f"{pb:<7.1f} | {algo:<5} | "
                    f"{re:<10.4f}{f1:<10.4f}{f2:<10.4f}"
                    f"{s1:<10.4f}{s2:<10.4f}{bt:<10.4f}{ne:<10.4f}"
                )
            print(line)

    print("=" * 160)


# ============================================================
# 4. Main Entry Point
# ============================================================
if __name__ == "__main__":
    # Run the comprehensive evaluation
    results = evaluate_re_vs_p_both(
        Fmax_fixed=0.6,
        p_only_fixed=0.9,
        V_dpp=100,
        steps=120000,
        burn_in=20000,
    )

    # Save results and print summary
    visualize_and_save(results, save_path="result/re_vs_p_both.mat")