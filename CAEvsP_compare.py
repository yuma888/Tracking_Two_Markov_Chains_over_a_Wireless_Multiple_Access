import numpy as np
from itertools import product
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

# ---------------- Global Fixed Parameters ----------------
rng_seed = 123
np.random.seed(rng_seed)

# Algorithm parameters
RVI_TOL = 1e-2
RVI_MAX_IT = 5000
LAMBDA_MAX = 50
BISEC_TOL = 1e-3
MAX_INNER = 50
MAX_OUTER = 50

# Simulation configuration
SIMULATION_TIMES = 10  # Number of simulations per policy run
SIMULATION_BURN_IN = 2000  # Burn-in steps
SIMULATION_STEPS = 50000  # Effective steps per simulation

# ---------------- Global Configuration for Analysis ----------------
FMAX_FIXED = 0.5  # Fixed frequency constraint
V_DPP = 100  # DPP weight parameter
P_RANGE = np.arange(0.1, 1.01, 0.1)  # Channel success rate scan range


# ============================================================
# 1. Two-Source MPR-CMDP Model Class
# ============================================================
class TwoSourceMPRCMDP:
    def __init__(
            self,
            Q1=None,
            Q2=None,
            delta1=None,
            delta2=None,
            p1_only=0.9,
            p2_only=0.9,
            p1_both=0.5,
            p2_both=0.5,
            Fmax1=0.5,
            Fmax2=0.5,
    ):
        # Source Markov chains
        self.Q1 = Q1 if Q1 is not None else np.array([[0.8, 0.2],
                                                      [0.2, 0.8]])
        self.Q2 = Q2 if Q2 is not None else np.array([[0.2, 0.8],
                                                      [0.8, 0.2]])

        # CAE cost matrices
        self.delta1 = delta1 if delta1 is not None else np.array([[0.0, 10.0],
                                                                  [30.0, 0.0]])
        self.delta2 = delta2 if delta2 is not None else np.array([[0.0, 10.0],
                                                                  [30.0, 0.0]])

        # MPR channel success probabilities
        self.p1_only = p1_only
        self.p2_only = p2_only
        self.p1_both = p1_both
        self.p2_both = p2_both

        # Constraints F1<=Fmax1, F2<=Fmax2
        self.Fmax = np.array([Fmax1, Fmax2], dtype=float)

        # States: (x1, x2, xhat1, xhat2)
        self.states = list(product([0, 1], [0, 1], [0, 1], [0, 1]))
        self.num_states = len(self.states)
        self.state_index = {s: i for i, s in enumerate(self.states)}

        # Actions: (a1, a2), ai∈{0,1,2}
        self.actions = list(product([0, 1, 2], [0, 1, 2]))
        self.num_actions = len(self.actions)
        self.action_index = {a: i for i, a in enumerate(self.actions)}

        # Indicator functions: whether a joint action is transmitted by S1 / S2
        self.f1_a = np.array([1 if a[0] != 0 else 0 for a in self.actions], dtype=float)
        self.f2_a = np.array([1 if a[1] != 0 else 0 for a in self.actions], dtype=float)

        # Transition kernel & expected CAE
        self.P = self._build_transition_kernel()
        self.c_sa = self._precompute_expected_cae()

    # ----------------- Channel Success Probabilities -----------------
    def channel_success_probs(self, a1, a2):
        if a1 == 0 and a2 == 0:
            return 0.0, 0.0
        if a1 != 0 and a2 == 0:
            return self.p1_only, 0.0
        if a1 == 0 and a2 != 0:
            return 0.0, self.p2_only
        # Both transmit
        return self.p1_both, self.p2_both

    # ----------------- Build Transition Kernel P[s,a,s'] -----------------
    def _build_transition_kernel(self):
        S, A = self.num_states, self.num_actions
        P = np.zeros((S, A, S), dtype=float)

        for s_idx, s in enumerate(self.states):
            x1, x2, xhat1, xhat2 = s

            for a_idx, (a1, a2) in enumerate(self.actions):
                p1s, p2s = self.channel_success_probs(a1, a2)

                for k1 in [0, 1]:
                    for k2 in [0, 1]:
                        pQ = self.Q1[x1, k1] * self.Q2[x2, k2]

                        outcomes = [
                            (1, 1, p1s * p2s),
                            (1, 0, p1s * (1 - p2s)),
                            (0, 1, (1 - p1s) * p2s),
                            (0, 0, (1 - p1s) * (1 - p2s)),
                        ]

                        for s1_succ, s2_succ, pchan in outcomes:
                            if pchan == 0:
                                continue

                            succ1 = (a1 == 1 and s1_succ == 1) or (a2 == 1 and s2_succ == 1)
                            succ2 = (a1 == 2 and s1_succ == 1) or (a2 == 2 and s2_succ == 1)

                            xhat1p = k1 if succ1 else xhat1
                            xhat2p = k2 if succ2 else xhat2

                            sp = (k1, k2, xhat1p, xhat2p)
                            sp_idx = self.state_index[sp]
                            P[s_idx, a_idx, sp_idx] += pQ * pchan

        # Normalize each (s,a) row to avoid numerical errors
        for s in range(S):
            for a in range(A):
                total = P[s, a, :].sum()
                if total > 0:
                    P[s, a, :] /= total

        return P

    # ----------------- Precompute Expected CAE c(s,a) -----------------
    def _precompute_expected_cae(self):
        S, A = self.num_states, self.num_actions
        c_sa = np.zeros((S, A), dtype=float)

        for s in range(S):
            for a in range(A):
                total = 0.0
                for sp_idx, prob in enumerate(self.P[s, a, :]):
                    if prob == 0:
                        continue
                    x1p, x2p, xhat1p, xhat2p = self.states[sp_idx]
                    cae = self.delta1[x1p, xhat1p] + self.delta2[x2p, xhat2p]
                    total += prob * cae
                c_sa[s, a] = total

        return c_sa

    # ----------------- Stationary Distribution Calculation -----------------
    def stationary_distribution(self, Ppi):
        pi = np.ones(self.num_states) / self.num_states
        for _ in range(20000):
            pi2 = pi @ Ppi
            if np.max(np.abs(pi2 - pi)) < 1e-12:
                break
            pi = pi2
        pi = np.maximum(pi, 0.0)
        s = pi.sum()
        if s <= 0:
            return np.ones(self.num_states) / self.num_states
        return pi / s

    # ----------------- RVI -----------------
    def RVI(self, lambda1, lambda2):
        """Relative Value Iteration for solving average cost MDP (Dual Bisection Search)"""
        l_sa = self.c_sa + lambda1 * self.f1_a[np.newaxis, :] + lambda2 * self.f2_a[np.newaxis, :]
        v = np.zeros(self.num_states)

        for _ in range(RVI_MAX_IT):
            v_new = np.zeros_like(v)
            for s in range(self.num_states):
                vals = l_sa[s] + self.P[s] @ v
                v_new[s] = np.min(vals)
            ref = v_new[0]
            v_new -= ref
            if np.max(np.abs(v_new - v)) < RVI_TOL:
                break
            v = v_new

        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            vals = l_sa[s] + self.P[s] @ v
            policy[s] = np.argmin(vals)

        Ppi = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            Ppi[s] = self.P[s, policy[s]]

        pi = self.stationary_distribution(Ppi)

        CAE = sum(pi[s] * self.c_sa[s, policy[s]] for s in range(self.num_states))
        F1 = sum(pi[s] * self.f1_a[policy[s]] for s in range(self.num_states))
        F2 = sum(pi[s] * self.f2_a[policy[s]] for s in range(self.num_states))

        return policy, CAE, F1, F2

    # ----------------- Inner Bisection Search -----------------
    def inner_lambda2_search(self, lambda1, F2_max):
        """Inner bisection search: fix λ1, search λ2, and mix between λ2^- and λ2^+ policies to hit F2_max"""
        lam2_low = 0.0
        lam2_high = LAMBDA_MAX

        pol_low, CAE_low, F1_low, F2_low = self.RVI(lambda1, lam2_low)
        pol_high, CAE_high, F1_high, F2_high = self.RVI(lambda1, lam2_high)

        for _ in range(MAX_INNER):
            lam2_mid = 0.5 * (lam2_low + lam2_high)
            pol_mid, CAE_mid, F1_mid, F2_mid = self.RVI(lambda1, lam2_mid)

            if F2_mid > F2_max:
                lam2_low = lam2_mid
                pol_low, CAE_low, F1_low, F2_low = pol_mid, CAE_mid, F1_mid, F2_mid
            else:
                lam2_high = lam2_mid
                pol_high, CAE_high, F1_high, F2_high = pol_mid, CAE_mid, F1_mid, F2_mid

            if abs(lam2_high - lam2_low) < BISEC_TOL:
                break

        # Mix policies at both ends of λ2 to accurately hit F2_max
        denom = F2_high - F2_low
        if abs(denom) < 1e-12:
            mu = 0.5
        else:
            mu = (F2_high - F2_max) / denom
        mu = float(np.clip(mu, 0.0, 1.0))

        CAE_mix = mu * CAE_low + (1 - mu) * CAE_high
        F1_mix = mu * F1_low + (1 - mu) * F1_high
        F2_mix = mu * F2_low + (1 - mu) * F2_high

        # Build the mixed random policy π_λ2(s,a) from inner mixing
        S, A = self.num_states, self.num_actions
        pi_mix_lambda2 = np.zeros((S, A))
        for s in range(S):
            pi_mix_lambda2[s, pol_low[s]] += mu
            pi_mix_lambda2[s, pol_high[s]] += (1 - mu)

        return {
            "CAE": CAE_mix,
            "F1": F1_mix,
            "F2": F2_mix,
            "lam2_low": lam2_low,
            "lam2_high": lam2_high,
            "pol_low": pol_low,
            "pol_high": pol_high,
            "mu": mu,
            "pi_mix": pi_mix_lambda2
        }

    # ----------------- Outer Bisection Search -----------------
    def outer_lambda1_search(self, F1_max, F2_max):
        """
        Outer bisection search: call inner_lambda2_search at λ1^- and λ1^+ to get
        two "inner-mixed" policies π_λ1^- and π_λ1^+, then linearly mix between them
        to precisely hit F1_max.
        """

        # 1) First check λ1 = 0 case
        res_low = self.inner_lambda2_search(0.0, F2_max)
        F1_low = res_low["F1"]

        # If λ1 = 0 already satisfies F1 constraint (and F2 is already at F2_max from inner mixing),
        # then constraint is inactive, return λ1=0 solution directly
        if F1_low <= F1_max + BISEC_TOL:
            return {
                "CAE": res_low["CAE"],
                "F1": res_low["F1"],
                "F2": res_low["F2"],
                "lam1": 0.0,
                "lam2_low": res_low["lam2_low"],
                "lam2_high": res_low["lam2_high"],
                "mu_inner": res_low["mu"],
                "nu_outer": 1.0,  # Only use low
                "pi_mix": res_low["pi_mix"]
            }

        # 2) Find λ1_high such that F1_high <= F1_max
        lam1_low = 0.0
        lam1_high = 1.0
        res_high = None
        for _ in range(40):
            res_high = self.inner_lambda2_search(lam1_high, F2_max)
            F1_high = res_high["F1"]
            if F1_high <= F1_max + BISEC_TOL:
                break
            lam1_high *= 2.0
            if lam1_high > LAMBDA_MAX:
                lam1_high = LAMBDA_MAX
                break

        if res_high is None:
            res_high = self.inner_lambda2_search(lam1_high, F2_max)
        F1_high = res_high["F1"]

        # If even at λ1_high, F1_high > F1_max, constraint is very tight or numerical issue
        # Return solution at λ1_high directly (smallest F1 we can find)
        if F1_high > F1_max + BISEC_TOL:
            return {
                "CAE": res_high["CAE"],
                "F1": res_high["F1"],
                "F2": res_high["F2"],
                "lam1": lam1_high,
                "lam2_low": res_high["lam2_low"],
                "lam2_high": res_high["lam2_high"],
                "mu_inner": res_high["mu"],
                "nu_outer": 1.0,
                "pi_mix": res_high["pi_mix"]
            }

        # Now F1_low > F1_max >= F1_high, meaning (λ1_low, λ1_high) crosses F1_max
        # 3) Formal outer bisection on λ1
        for _ in range(MAX_OUTER):
            lam1_mid = 0.5 * (lam1_low + lam1_high)
            res_mid = self.inner_lambda2_search(lam1_mid, F2_max)
            F1_mid = res_mid["F1"]

            if F1_mid > F1_max:  # Frequency still too high, need larger λ1
                lam1_low = lam1_mid
                res_low = res_mid
            else:  # F1_mid <= F1_max
                lam1_high = lam1_mid
                res_high = res_mid

            if abs(lam1_high - lam1_low) < BISEC_TOL:
                break

        # 4) Mix between the two "inner-mixed policies" at λ1_low and λ1_high to precisely hit F1_max
        C_low, F1_low, F2_low, pi_low = res_low["CAE"], res_low["F1"], res_low["F2"], res_low["pi_mix"]
        C_high, F1_high, F2_high, pi_high = res_high["CAE"], res_high["F1"], res_high["F2"], res_high["pi_mix"]

        denom = F1_low - F1_high
        if abs(denom) < 1e-12:
            nu = 0.5
        else:
            # Solve nu·F1_low + (1-nu)·F1_high = F1_max
            nu = (F1_max - F1_high) / denom
        nu = float(np.clip(nu, 0.0, 1.0))

        CAE_final = nu * C_low + (1 - nu) * C_high
        F1_final = nu * F1_low + (1 - nu) * F1_high
        F2_final = nu * F2_low + (1 - nu) * F2_high
        pi_final = nu * pi_low + (1 - nu) * pi_high

        # Take "expected value" of λ1 (just for reference, actual policy comes from mixing)
        lam1_star = nu * lam1_low + (1 - nu) * lam1_high

        return {
            "CAE": CAE_final,
            "F1": F1_final,
            "F2": F2_final,
            "lam1": lam1_star,
            "lam2_low": (res_low["lam2_low"], res_high["lam2_low"]),
            "lam2_high": (res_low["lam2_high"], res_high["lam2_high"]),
            "mu_inner": (res_low["mu"], res_high["mu"]),
            "nu_outer": nu,
            "pi_mix": pi_final
        }

    # ----------------- Dual Bisection Search Main Entry -----------------
    def double_bisection_rvi(self):
        """Double bisection search RVI for solving CMDP (inner λ2 + outer λ1 both with mixing)"""
        start_time = time.time()
        res = self.outer_lambda1_search(self.Fmax[0], self.Fmax[1])
        elapsed_time = time.time() - start_time

        res["compute_time"] = elapsed_time
        # Compatibility with field names used in main() later
        return {
            'CAE': res["CAE"],
            'F1': res["F1"],
            'F2': res["F2"],
            'lambda1': res["lam1"],
            'lambda2_range': res["lam2_low"],  # Put some info here, not important
            'mix_coeff': {'mu_inner': res["mu_inner"], 'nu_outer': res["nu_outer"]},
            'mix_policy': res["pi_mix"],
            'compute_time': elapsed_time
        }

    # ----------------- Supplement: simulate_transmission method (adapt to original simulation function) -----------------
    def simulate_transmission(self, s_idx, action):
        a_idx = self.action_index[action]
        next_s_idx = np.random.choice(self.num_states, p=self.P[s_idx, a_idx])
        actual_cae = self.c_sa[s_idx, a_idx]
        return next_s_idx, actual_cae


# ============================================================
# 2. Utility Functions: Stationary Distribution, Policy Evaluation, Multiple Simulations for Average
# ============================================================
def stationary(Ppi, tol=1e-12, max_iter=10000):
    S = Ppi.shape[0]
    mu = np.ones(S) / S
    for _ in range(max_iter):
        mu_next = mu @ Ppi
        if np.max(np.abs(mu_next - mu)) < tol:
            break
        mu = mu_next
    mu = np.maximum(mu, 0.0)
    s = mu.sum()
    if s <= 0:
        return np.ones(S) / S
    return mu / s


def eval_policy_theoretical(cmdp: TwoSourceMPRCMDP, pi):
    S, A = cmdp.num_states, cmdp.num_actions
    Ppi = np.zeros((S, S), dtype=float)
    for s in range(S):
        for a in range(A):
            Ppi[s, :] += pi[s, a] * cmdp.P[s, a, :]

    mu = stationary(Ppi)

    C = 0.0
    F1 = 0.0
    F2 = 0.0
    for s in range(S):
        for a in range(A):
            C += mu[s] * pi[s, a] * cmdp.c_sa[s, a]
            F1 += mu[s] * pi[s, a] * cmdp.f1_a[a]
            F2 += mu[s] * pi[s, a] * cmdp.f2_a[a]

    return mu, float(C), float(F1), float(F2)


def simulate_policy_single(cmdp: TwoSourceMPRCMDP, pi, mu0=None, seed=None):
    """Single simulation (independent seed)"""
    if seed is not None:
        np.random.seed(seed)

    S, A = cmdp.num_states, cmdp.num_actions

    if mu0 is None:
        s = 0
    else:
        s = np.random.choice(S, p=mu0)

    C_sum = 0.0
    F1_sum = 0.0
    F2_sum = 0.0
    count = 0

    for t in range(SIMULATION_STEPS + SIMULATION_BURN_IN):
        a = np.random.choice(A, p=pi[s])
        a1, a2 = cmdp.actions[a]

        if t >= SIMULATION_BURN_IN:
            C_sum += cmdp.c_sa[s, a]
            F1_sum += (1 if a1 != 0 else 0)
            F2_sum += (1 if a2 != 0 else 0)
            count += 1

        s = np.random.choice(S, p=cmdp.P[s, a])

    return C_sum / count, F1_sum / count, F2_sum / count


def simulate_policy_multiple(cmdp: TwoSourceMPRCMDP, pi, mu0=None, times=SIMULATION_TIMES):
    """Multiple simulations for average, return (mean, standard deviation)"""
    c_list, f1_list, f2_list = [], [], []

    # Use different random seeds for each simulation to ensure independence
    base_seed = np.random.randint(0, 1000000)

    for i in range(times):
        c, f1, f2 = simulate_policy_single(cmdp, pi, mu0, seed=base_seed + i)
        c_list.append(c)
        f1_list.append(f1)
        f2_list.append(f2)

    # Calculate mean and standard deviation
    c_mean = np.mean(c_list)
    c_std = np.std(c_list)
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)
    f2_mean = np.mean(f2_list)
    f2_std = np.std(f2_list)

    return (c_mean, c_std), (f1_mean, f1_std), (f2_mean, f2_std)


# ============================================================
# 3. Occupation-measure LP for Solving CMDP
# ============================================================
def solve_cmdp_via_lp(cmdp: TwoSourceMPRCMDP):
    """
    Solve the Constrained Markov Decision Process (CMDP) using the occupation-measure Linear Programming (LP) approach.

    This method transforms the CMDP into a finite-dimensional LP by leveraging the occupation measure, which
    fully characterizes the long-term behavior of stationary policies. The LP minimizes the long-term average
    Cost of Actuation Error (CAE) while satisfying per-sensor transmission frequency constraints.

    Args:
        cmdp (TwoSourceMPRCMDP): Instance of the two-source MPR-CMDP model, containing system dynamics,
                                cost matrices, channel success probabilities, and transmission constraints.

    Returns:
        tuple:
            x (np.ndarray): Optimal occupation measure (S×A), where x[s,a] is the steady-state fraction of time
                            the system is in state s and takes action a.
            pi (np.ndarray): Optimal stationary randomized policy (S×A), where pi[s,a] is the probability of
                            taking action a in state s.
            mu_s (np.ndarray): Stationary distribution of states (S×1), where mu_s[s] is the steady-state
                              probability of being in state s.
            C_opt (float): Minimum achievable long-term average CAE.
            F1_opt (float): Actual long-term average transmission frequency of Sensor 1 (satisfies F1 ≤ Fmax1).
            F2_opt (float): Actual long-term average transmission frequency of Sensor 2 (satisfies F2 ≤ Fmax2).
    """
    # Get the number of states (S) and actions (A) from the CMDP model
    S, A = cmdp.num_states, cmdp.num_actions
    # Total number of optimization variables (state-action pairs)
    N = S * A

    def idx(s, a):
        """
        Map state-action pair (s, a) to a 1-dimensional index for LP vectorization.

        Args:
            s (int): State index (0 to S-1).
            a (int): Action index (0 to A-1).

        Returns:
            int: 1D index corresponding to (s, a) (0 to N-1).
        """
        return s * A + a

    # Initialize vectors for LP objective and constraints
    # c_vec: Objective function coefficients (1×N) – one-step expected CAE for each (s,a) pair
    c_vec = np.zeros(N)
    # F1_vec/F2_vec: Transmission frequency constraint coefficients (1×N) – indicator if action a triggers transmission
    F1_vec = np.zeros(N)
    F2_vec = np.zeros(N)

    # Populate objective and constraint vectors with values from the CMDP model
    for s in range(S):
        for a in range(A):
            # Convert (s,a) to 1D index
            i = idx(s, a)
            # Objective coefficient: one-step expected CAE for state s and action a
            c_vec[i] = cmdp.c_sa[s, a]
            # Constraint coefficient: 1 if action a makes Sensor 1 transmit, 0 otherwise
            F1_vec[i] = cmdp.f1_a[a]
            # Constraint coefficient: 1 if action a makes Sensor 2 transmit, 0 otherwise
            F2_vec[i] = cmdp.f2_a[a]

    # Initialize equality constraint matrix (A_eq) and right-hand side (b_eq)
    # A_eq: (S+1)×N matrix – S flow-balance constraints + 1 normalization constraint
    A_eq = np.zeros((S + 1, N))
    b_eq = np.zeros(S + 1)

    # Populate flow-balance constraints (first S rows of A_eq)
    # Flow-balance: For each target state sp, incoming occupation measure = outgoing occupation measure
    for sp in range(S):
        for s in range(S):
            for a in range(A):
                # Coefficient for (s,a) in the flow-balance equation for sp:
                # +1 if s == sp (incoming to sp), -P(sp|s,a) (outgoing from s via a to sp)
                A_eq[sp, idx(s, a)] += (1.0 if s == sp else 0.0) - cmdp.P[s, a, sp]
        # Right-hand side of flow-balance constraint: 0 (equality between incoming and outgoing)
        b_eq[sp] = 0.0

    # Populate normalization constraint (last row of A_eq)
    # Normalization: Sum of all occupation measures = 1 (probability distribution property)
    A_eq[S, :] = 1.0
    b_eq[S] = 1.0

    # Initialize inequality constraint matrix (A_ub) and right-hand side (b_ub)
    # A_ub: 2×N matrix – transmission frequency constraints for Sensor 1 and Sensor 2
    A_ub = np.vstack([F1_vec, F2_vec])
    # b_ub: Maximum allowable transmission frequencies for each sensor (from CMDP's Fmax)
    b_ub = cmdp.Fmax.copy()

    # Variable bounds: Occupation measures are non-negative (probability can't be negative)
    bounds = [(0.0, None) for _ in range(N)]

    # Solve the LP problem using the HiGHS solver (efficient for linear programming)
    # Objective: Minimize c_vec^T * x (long-term average CAE)
    # Subject to: A_eq * x = b_eq (flow-balance + normalization)
    #             A_ub * x ≤ b_ub (transmission frequency constraints)
    #             x ≥ 0 (non-negativity of occupation measures)
    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    # Check if LP solving succeeded; raise error if not
    if not res.success:
        raise RuntimeError("LP solve failed: " + res.message)

    # Reshape the 1D optimal solution back to S×A matrix (occupation measure)
    x = res.x.reshape(S, A)

    # Compute stationary state distribution: sum occupation measures over all actions for each state
    mu_s = x.sum(axis=1)

    # Recover optimal stationary randomized policy: pi[s,a] = x[s,a] / mu_s[s]
    # If mu_s[s] is negligible (≈0), use uniform random policy for state s (avoids division by zero)
    pi = np.zeros((S, A))
    for s in range(S):
        if mu_s[s] > 1e-12:
            # Probability of action a in state s: normalized occupation measure
            pi[s, :] = x[s, :] / mu_s[s]
        else:
            # Uniform random policy for states with negligible stationary probability
            pi[s, :] = 1.0 / A

    # Calculate optimal performance metrics
    # Long-term average CAE: sum over all (s,a) of (one-step CAE) * (occupation measure)
    C_opt = float((cmdp.c_sa * x).sum())
    # Long-term average transmission frequency of Sensor 1: sum over (s,a) of (transmission indicator) * (occupation measure)
    F1_opt = float((cmdp.f1_a[np.newaxis, :] * x).sum())
    # Long-term average transmission frequency of Sensor 2: same logic as Sensor 1
    F2_opt = float((cmdp.f2_a[np.newaxis, :] * x).sum())

    return x, pi, mu_s, C_opt, F1_opt, F2_opt


# ============================================================
# 4. Fixed Probability Strategy (FPS) Policy
# ============================================================
def build_fixed_policy(cmdp: TwoSourceMPRCMDP):
    A = cmdp.num_actions
    S = cmdp.num_states
    F1_max, F2_max = cmdp.Fmax

    p1 = {0: 1 - F1_max, 1: F1_max / 2, 2: F1_max / 2}
    p2 = {0: 1 - F2_max, 1: F2_max / 2, 2: F2_max / 2}

    pi_a = np.zeros(A, dtype=float)
    for idx, (a1, a2) in enumerate(cmdp.actions):
        pi_a[idx] = p1[a1] * p2[a2]

    pi = np.tile(pi_a, (S, 1))
    return pi


# ============================================================
# 5. DPP Algorithm Class
# ============================================================
class LowComplexityDPP:
    def __init__(self, cmdp, V1=100, V2=100):
        self.cmdp = cmdp
        self.V = np.array([V1, V2])
        self.Z = np.zeros(2)  # Virtual queue
        self.reset()

    def reset(self):
        self.Z = np.zeros(2)
        self.total_cae = 0.0
        self.transmission_counts = np.zeros(2)
        self.time_slot = 0
        self.current_state_idx = 0
        self.current_state = self.cmdp.states[0]

    def compute_dpp_objective(self, state_idx, action_idx):
        expected_cae = self.cmdp.c_sa[state_idx, action_idx]
        f1 = self.cmdp.f1_a[action_idx]
        f2 = self.cmdp.f2_a[action_idx]
        return self.V[0] * expected_cae + self.Z[0] * f1 + self.Z[1] * f2

    def select_action(self):
        best_value = float('inf')
        best_action_idx = 0
        for a_idx in range(self.cmdp.num_actions):
            val = self.compute_dpp_objective(self.current_state_idx, a_idx)
            if val < best_value:
                best_value = val
                best_action_idx = a_idx
        return best_action_idx, self.cmdp.actions[best_action_idx]

    def update_virtual_queues(self, action):
        a1, a2 = action
        f1 = 1 if a1 != 0 else 0
        f2 = 1 if a2 != 0 else 0
        self.Z[0] = max(self.Z[0] - self.cmdp.Fmax[0] + f1, 0)
        self.Z[1] = max(self.Z[1] - self.cmdp.Fmax[1] + f2, 0)

    def simulate_transmission(self, action):
        a1, a2 = action
        s_idx = self.current_state_idx
        a_idx = self.cmdp.action_index[action]
        next_state_idx = np.random.choice(self.cmdp.num_states, p=self.cmdp.P[s_idx, a_idx])
        actual_cae = self.cmdp.c_sa[s_idx, a_idx]
        self.current_state_idx = next_state_idx
        self.current_state = self.cmdp.states[next_state_idx]
        return next_state_idx, actual_cae

    def run_simulation(self, num_steps=50000, burn_in=2000, count_actions=False):
        self.reset()
        self.stats = {
            "single1": 0, "single2": 0, "both": 0, "none": 0,
            "s1_src1": 0, "s1_src2": 0, "s2_src1": 0, "s2_src2": 0
        }

        # Burn-in phase
        for _ in range(burn_in):
            a_idx, action = self.select_action()
            self.simulate_transmission(action)
            self.update_virtual_queues(action)

        # Formal simulation
        for _ in range(num_steps):
            a_idx, action = self.select_action()
            next_state_idx, cae = self.simulate_transmission(action)
            self.update_virtual_queues(action)

            a1, a2 = action
            f1 = 1 if a1 != 0 else 0
            f2 = 1 if a2 != 0 else 0

            # Accumulate statistics
            self.total_cae += cae
            self.transmission_counts[0] += f1
            self.transmission_counts[1] += f2

            # Detailed action statistics
            if count_actions:
                if f1 and not f2:
                    self.stats["single1"] += 1
                elif f2 and not f1:
                    self.stats["single2"] += 1
                elif f1 and f2:
                    self.stats["both"] += 1
                else:
                    self.stats["none"] += 1

                if a1 == 1:
                    self.stats["s1_src1"] += 1
                elif a1 == 2:
                    self.stats["s1_src2"] += 1
                if a2 == 1:
                    self.stats["s2_src1"] += 1
                elif a2 == 2:
                    self.stats["s2_src2"] += 1

        # Normalization
        avg_cae = self.total_cae / num_steps
        avg_freq = self.transmission_counts / num_steps
        if count_actions:
            for k in self.stats:
                self.stats[k] /= num_steps

        return avg_cae, avg_freq


# ---------------- Enhanced DPP (Complete Statistics) ----------------
class EnhancedDPP(LowComplexityDPP):
    def run_simulation(self, num_steps=50000, burn_in=2000, count_actions=True):
        return super().run_simulation(num_steps, burn_in, count_actions)


# ============================================================
# 6. Enhanced Simulation Function (Statistics for Fast/Slow Source Scheduling Frequency)
# ============================================================
def simulate_policy_mc_enhanced(cmdp, policy, num_steps=50000, burn_in=2000):
    """Monte-Carlo Simulation (Statistics for Fast/Slow Source Scheduling, Transmission Modes)"""
    S, A = cmdp.num_states, cmdp.num_actions
    state = 0
    total_cae = 0.0
    total_f1 = 0
    total_f2 = 0
    stats = {
        "single1": 0, "single2": 0, "both": 0, "none": 0,
        "s1_src1": 0, "s1_src2": 0, "s2_src1": 0, "s2_src2": 0
    }

    # Burn-in phase
    for _ in range(burn_in):
        a_idx = np.random.choice(A, p=policy[state])
        state = np.random.choice(S, p=cmdp.P[state, a_idx])

    # Formal simulation
    for _ in range(num_steps):
        a_idx = np.random.choice(A, p=policy[state])
        a1, a2 = cmdp.actions[a_idx]
        next_state = np.random.choice(S, p=cmdp.P[state, a_idx])

        # Accumulate CAE and frequency
        total_cae += cmdp.c_sa[state, a_idx]
        f1 = 1 if a1 != 0 else 0
        f2 = 1 if a2 != 0 else 0
        total_f1 += f1
        total_f2 += f2

        # Action statistics
        if f1 and not f2:
            stats["single1"] += 1
        elif f2 and not f1:
            stats["single2"] += 1
        elif f1 and f2:
            stats["both"] += 1
        else:
            stats["none"] += 1

        if a1 == 1:
            stats["s1_src1"] += 1  # Sensor 1 transmits slow source
        elif a1 == 2:
            stats["s1_src2"] += 1  # Sensor 1 transmits fast source
        if a2 == 1:
            stats["s2_src1"] += 1  # Sensor 2 transmits slow source
        elif a2 == 2:
            stats["s2_src2"] += 1  # Sensor 2 transmits fast source

        state = next_state

    # Normalization
    cnt = num_steps
    result = {
        "avg_cae": total_cae / cnt,
        "freq1": total_f1 / cnt,
        "freq2": total_f2 / cnt
    }
    for k in stats:
        result[k] = stats[k] / cnt
    return result


# ============================================================
# 7. Main Evaluation Function: Analyze Fast/Slow Source Scheduling Frequency, Transmission Modes, Actual Frequency & CAE
# ============================================================
def evaluate_source_dynamics():
    """Evaluate Fast/Slow Source Scheduling Frequency, Transmission Modes, Actual Frequency & CAE"""
    results = []
    # Fixed source transition probabilities (fast source Q1, slow source Q2)
    Q1_fast = np.array([[0.8, 0.2],
                        [0.2, 0.8]])  # Slow source: low transition probability
    Q2_slow = np.array([[0.2, 0.8],
                        [0.8, 0.2]])  # Fast source: high transition probability

    print(f"\n=== Evaluation Configuration ===")
    print(f"Fixed Fmax: {FMAX_FIXED} | Slow Source Q1: {Q1_fast} | Fast Source Q2: {Q2_slow}")
    print(f"Simulation Steps: {SIMULATION_STEPS} | Burn-in Steps: {SIMULATION_BURN_IN}")
    print(f"Channel Success Probability Range: {P_RANGE}\n")

    for p in P_RANGE:
        print(f">>> Channel Success Probability p = {p:.1f}")
        row = {"p": float(p)}

        # Build CMDP (fixed fast/slow sources, scan channel success rates)
        cmdp = TwoSourceMPRCMDP(
            Q1=Q1_fast, Q2=Q2_slow,
            p1_only=p, p2_only=p, p1_both=p / 2, p2_both=p / 2,
            Fmax1=FMAX_FIXED, Fmax2=FMAX_FIXED
        )

        # 1. LP Algorithm
        try:
            _, pi_lp, _, _, _, _ = solve_cmdp_via_lp(cmdp)
            row["LP"] = simulate_policy_mc_enhanced(cmdp, pi_lp, SIMULATION_STEPS, SIMULATION_BURN_IN)
            print(f"  LP: CAE={row['LP']['avg_cae']:.4f}, F1={row['LP']['freq1']:.4f}, F2={row['LP']['freq2']:.4f}")
        except Exception as e:
            row["LP"] = None
            print(f"  LP: Failed - {e}")

        # 2. RVI Algorithm
        try:
            rvi_res = cmdp.double_bisection_rvi()
            pi_rvi = rvi_res["mix_policy"]
            row["RVI"] = simulate_policy_mc_enhanced(cmdp, pi_rvi, SIMULATION_STEPS, SIMULATION_BURN_IN)
            print(f"  RVI: CAE={row['RVI']['avg_cae']:.4f}, F1={row['RVI']['freq1']:.4f}, F2={row['RVI']['freq2']:.4f}")
        except Exception as e:
            row["RVI"] = None
            print(f"  RVI: Failed - {e}")

        # 3. FPS Fixed Policy
        try:
            pi_fps = build_fixed_policy(cmdp)
            row["FPS"] = simulate_policy_mc_enhanced(cmdp, pi_fps, SIMULATION_STEPS, SIMULATION_BURN_IN)
            print(f"  FPS: CAE={row['FPS']['avg_cae']:.4f}, F1={row['FPS']['freq1']:.4f}, F2={row['FPS']['freq2']:.4f}")
        except Exception as e:
            row["FPS"] = None
            print(f"  FPS: Failed - {e}")

        # 4. DPP Algorithm
        try:
            dpp = EnhancedDPP(cmdp, V1=V_DPP, V2=V_DPP)
            avg_cae, (freq1, freq2) = dpp.run_simulation(SIMULATION_STEPS, SIMULATION_BURN_IN)
            row["DPP"] = {
                "avg_cae": avg_cae,
                "freq1": freq1,
                "freq2": freq2,
                "single1": dpp.stats["single1"],
                "single2": dpp.stats["single2"],
                "both": dpp.stats["both"],
                "none": dpp.stats["none"],
                "s1_src1": dpp.stats["s1_src1"],  # Sensor 1 transmits slow source
                "s1_src2": dpp.stats["s1_src2"],  # Sensor 1 transmits fast source
                "s2_src1": dpp.stats["s2_src1"],  # Sensor 2 transmits slow source
                "s2_src2": dpp.stats["s2_src2"]  # Sensor 2 transmits fast source
            }
            print(f"  DPP: CAE={avg_cae:.4f}, F1={freq1:.4f}, F2={freq2:.4f}")
        except Exception as e:
            row["DPP"] = None
            print(f"  DPP: Failed - {e}")

        results.append(row)
        print("-" * 100)

    return results


# ============================================================
# 9. Data Saving & Summary Output
# ============================================================
def save_results_flat(results):
    """Store results as fully flattened variables (one array per algorithm per metric)"""

    p_vals = np.array([r["p"] for r in results])
    algorithms = ["LP", "RVI", "FPS", "DPP"]
    metrics = ["avg_cae", "freq1", "freq2", "single1", "single2",
               "both", "none", "s1_src1", "s1_src2", "s2_src1", "s2_src2"]

    # Initialize empty arrays: one data array per algorithm × per metric
    flat_data = {"p": p_vals}

    for algo in algorithms:
        for m in metrics:
            arr = []
            for r in results:
                if r[algo] is None:
                    arr.append(np.nan)
                else:
                    arr.append(r[algo][m])
            flat_data[f"{algo}_{m}"] = np.array(arr)

    # Save as mat file
    try:
        import scipy.io as sio
        sio.savemat("result/cae_vs_p.mat", flat_data)
        print("\nFlattened results saved to flat_source_dynamics.mat")
    except:
        print("\nscipy.io not found, MAT file not saved")

    # Optional print
    print("\nSaved variable names:")
    for k in flat_data:
        print("  ", k)

    return flat_data


# ============================================================
# 10. New: Print Multi-Dimensional Performance Summary Table
# ============================================================
def print_performance_summary(results):
    """Print detailed performance summary table including all core metrics"""
    algorithms = ["LP", "RVI", "FPS", "DPP"]
    metrics = [
        ("avg_cae", "Average CAE"),
        ("freq1", "Sensor 1 Frequency"),
        ("freq2", "Sensor 2 Frequency"),
        ("single1", "Sensor 1 Only Transmission"),
        ("single2", "Sensor 2 Only Transmission"),
        ("both", "Dual Sensor Simultaneous Transmission"),
        ("s1_src1", "Sensor 1 - Slow Source"),
        ("s1_src2", "Sensor 1 - Fast Source"),
        ("s2_src1", "Sensor 2 - Slow Source"),
        ("s2_src2", "Sensor 2 - Fast Source")
    ]

    # Print header
    print("\n" + "=" * 180)
    print("Performance Summary Table (Grouped by Channel Success Probability p)")
    print("=" * 180)

    # Iterate over each p value and print metrics for all algorithms
    for row in results:
        p = row["p"]
        print(f"\n--- Channel Success Probability p = {p:.1f} ---")

        # Build table header
        table_header = f"{'Algorithm':<8}"
        for _, metric_name in metrics:
            table_header += f"| {metric_name:<12}"
        print(table_header)
        print("-" * 180)

        # Print metrics for each algorithm
        for algo in algorithms:
            if row[algo] is None:
                # Algorithm failed to run
                row_str = f"{algo:<8}"
                for _ in metrics:
                    row_str += f"| {'Failed':<12}"
                print(row_str)
            else:
                # Algorithm ran successfully, print specific values
                row_str = f"{algo:<8}"
                for metric_key, _ in metrics:
                    val = row[algo][metric_key]
                    row_str += f"| {val:<12.4f}"
                print(row_str)

    # Print average summary (by algorithm)
    print("\n" + "=" * 180)
    print("Average Performance Summary by Algorithm (Averaged over all p values)")
    print("=" * 180)

    # Calculate average metrics for each algorithm
    algo_avg = {}
    for algo in algorithms:
        algo_avg[algo] = {}
        valid_p_count = 0
        for metric_key, _ in metrics:
            values = []
            for row in results:
                if row[algo] is not None:
                    values.append(row[algo][metric_key])
            if values:
                algo_avg[algo][metric_key] = np.mean(values)
                valid_p_count = max(valid_p_count, len(values))
            else:
                algo_avg[algo][metric_key] = np.nan

    # Print average table
    avg_header = f"{'Algorithm':<8}"
    for _, metric_name in metrics:
        avg_header += f"| {metric_name:<12}"
    print(avg_header)
    print("-" * 180)

    for algo in algorithms:
        row_str = f"{algo:<8}"
        for metric_key, _ in metrics:
            val = algo_avg[algo][metric_key]
            if np.isnan(val):
                row_str += f"| {'No Data':<12}"
            else:
                row_str += f"| {val:<12.4f}"
        print(row_str)

    # Print optimal algorithm marker (minimum CAE)
    print("\n" + "=" * 180)
    print("Optimal Algorithm by CAE for Each p value (Smaller CAE is Better)")
    print("=" * 180)
    print(f"{'p value':<8} | {'Best Algorithm':<8} | {'Minimum CAE':<12} | {'Second Best Algorithm':<8} | {'Second Best CAE':<12}")
    print("-" * 180)

    for row in results:
        p = row["p"]
        cae_vals = {}
        for algo in algorithms:
            if row[algo] is not None:
                cae_vals[algo] = row[algo]["avg_cae"]

        if cae_vals:
            # Sort CAE values
            sorted_cae = sorted(cae_vals.items(), key=lambda x: x[1])
            best_algo, best_cae = sorted_cae[0]
            if len(sorted_cae) >= 2:
                second_algo, second_cae = sorted_cae[1]
            else:
                second_algo, second_cae = "-", "-"

            print(f"{p:.1f}      | {best_algo:<8} | {best_cae:<12.4f} | {second_algo:<8} | {second_cae:<12.4f}")
        else:
            print(f"{p:.1f}      | {'No Valid Algorithm':<8} | {'-':<12} | {'-':<8} | {'-':<12}")



# ============================================================
# Main Function
# ============================================================
if __name__ == "__main__":
    # Run fast/slow source scheduling analysis (core functionality)
    results = evaluate_source_dynamics()

    # Print detailed performance summary table (new)
    print_performance_summary(results)

    # Save data & output summary
    flat_data = save_results_flat(results)
