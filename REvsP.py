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
RVI_TOL = 1e-2          # Relative Value Iteration tolerance
RVI_MAX_IT = 5000       # Maximum iterations for RVI
LAMBDA_MAX = 50         # Maximum Lagrange multiplier value
BISEC_TOL = 1e-3        # Bisection search tolerance
MAX_INNER = 50          # Maximum iterations for inner bisection (lambda2)
MAX_OUTER = 50          # Maximum iterations for outer bisection (lambda1)

# New: Simulation configuration (customizable)
SIMULATION_TIMES = 10   # Number of simulation runs per policy (for averaging)
SIMULATION_BURN_IN = 2000  # Burn-in steps for simulation
SIMULATION_STEPS = 50000    # Effective steps per simulation run

# ---------------- Global Configuration for Analysis ----------------
FMAX_FIXED = 0.5        # Fixed frequency constraint
V_DPP = 100             # DPP weight parameter
P_RANGE = np.arange(0.1, 1.01, 0.1)  # Channel success probability sweep range


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

        # RE (Reconstruction Error) cost matrices
        self.delta1 = delta1 if delta1 is not None else np.array([[0, 1],
                                                                  [1, 0]])
        self.delta2 = delta2 if delta2 is not None else np.array([[0, 1],
                                                                  [1, 0]])

        # MPR channel success probabilities
        self.p1_only = p1_only    # Success prob of source 1 when transmitting alone
        self.p2_only = p2_only    # Success prob of source 2 when transmitting alone
        self.p1_both = p1_both    # Success prob of source 1 when both transmit
        self.p2_both = p2_both    # Success prob of source 2 when both transmit

        # Constraints: F1<=Fmax1, F2<=Fmax2
        self.Fmax = np.array([Fmax1, Fmax2], dtype=float)

        # State space: (x1, x2, xhat1, xhat2)
        # x1/x2: Actual states of source 1/2
        # xhat1/xhat2: Estimated states of source 1/2 at receiver
        self.states = list(product([0, 1], [0, 1], [0, 1], [0, 1]))
        self.num_states = len(self.states)
        self.state_index = {s: i for i, s in enumerate(self.states)}

        # Action space: (a1, a2), ai∈{0,1,2}
        # 0: no transmission, 1: transmit source 1, 2: transmit source 2
        self.actions = list(product([0, 1, 2], [0, 1, 2]))
        self.num_actions = len(self.actions)
        self.action_index = {a: i for i, a in enumerate(self.actions)}

        # Indicator functions: whether a joint action involves transmission from S1/S2
        self.f1_a = np.array([1 if a[0] != 0 else 0 for a in self.actions], dtype=float)
        self.f2_a = np.array([1 if a[1] != 0 else 0 for a in self.actions], dtype=float)

        # Transition kernel & expected RE
        self.P = self._build_transition_kernel()
        self.c_sa = self._precompute_expected_re()

    # ----------------- Channel Success Probabilities -----------------
    def channel_success_probs(self, a1, a2):
        """Calculate channel success probabilities for given actions"""
        if a1 == 0 and a2 == 0:
            return 0.0, 0.0
        if a1 != 0 and a2 == 0:
            return self.p1_only, 0.0
        if a1 == 0 and a2 != 0:
            return 0.0, self.p2_only
        # Both sources transmit
        return self.p1_both, self.p2_both

    # ----------------- Build Transition Kernel P[s,a,s'] -----------------
    def _build_transition_kernel(self):
        """Construct transition kernel P[s,a,s']: probability of transitioning to s' from s under action a"""
        S, A = self.num_states, self.num_actions
        P = np.zeros((S, A, S), dtype=float)

        for s_idx, s in enumerate(self.states):
            x1, x2, xhat1, xhat2 = s

            for a_idx, (a1, a2) in enumerate(self.actions):
                p1s, p2s = self.channel_success_probs(a1, a2)

                # Iterate over possible next actual states (k1, k2)
                for k1 in [0, 1]:
                    for k2 in [0, 1]:
                        # Probability of source state transition
                        pQ = self.Q1[x1, k1] * self.Q2[x2, k2]

                        # All possible channel outcomes
                        outcomes = [
                            (1, 1, p1s * p2s),       # Both channels succeed
                            (1, 0, p1s * (1 - p2s)), # Only source 1 channel succeeds
                            (0, 1, (1 - p1s) * p2s), # Only source 2 channel succeeds
                            (0, 0, (1 - p1s) * (1 - p2s)) # Both channels fail
                        ]

                        for s1_succ, s2_succ, pchan in outcomes:
                            if pchan == 0:
                                continue

                            # Determine if reconstruction is successful
                            succ1 = (a1 == 1 and s1_succ == 1) or (a2 == 1 and s2_succ == 1)
                            succ2 = (a1 == 2 and s1_succ == 1) or (a2 == 2 and s2_succ == 1)

                            # Update estimated states
                            xhat1p = k1 if succ1 else xhat1
                            xhat2p = k2 if succ2 else xhat2

                            # Next state and its index
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

    # ----------------- Precompute Expected RE c(s,a) -----------------
    def _precompute_expected_re(self):
        """Precompute expected reconstruction error for each (state, action) pair"""
        S, A = self.num_states, self.num_actions
        c_sa = np.zeros((S, A), dtype=float)

        for s in range(S):
            for a in range(A):
                total = 0.0
                for sp_idx, prob in enumerate(self.P[s, a, :]):
                    if prob == 0:
                        continue
                    x1p, x2p, xhat1p, xhat2p = self.states[sp_idx]
                    re = self.delta1[x1p, xhat1p] + self.delta2[x2p, xhat2p]
                    total += prob * re
                c_sa[s, a] = total

        return c_sa

    # ----------------- Stationary Distribution Calculation -----------------
    def stationary_distribution(self, Ppi):
        """Compute stationary distribution for transition matrix Ppi using power iteration"""
        pi = np.ones(self.num_states) / self.num_states
        for _ in range(20000):
            pi2 = pi @ Ppi
            if np.max(np.abs(pi2 - pi)) < 1e-12:
                break
            pi = pi2
        pi = np.maximum(pi, 0.0)  # Ensure non-negative
        s = pi.sum()
        if s <= 0:
            return np.ones(self.num_states) / self.num_states
        return pi / s

    # ----------------- Relative Value Iteration (RVI) -----------------
    def RVI(self, lambda1, lambda2):
        """Relative Value Iteration for average cost MDP (for double bisection search)"""
        # Augmented cost: RE + lambda1*F1 + lambda2*F2
        l_sa = self.c_sa + lambda1 * self.f1_a[np.newaxis, :] + lambda2 * self.f2_a[np.newaxis, :]
        v = np.zeros(self.num_states)

        for _ in range(RVI_MAX_IT):
            v_new = np.zeros_like(v)
            for s in range(self.num_states):
                # Bellman equation: v_new[s] = min_a [l(s,a) + sum_{s'} P(s,a,s')v(s')]
                vals = l_sa[s] + self.P[s] @ v
                v_new[s] = np.min(vals)
            # Normalize with reference state (first state) to prevent value drift
            ref = v_new[0]
            v_new -= ref
            # Check convergence
            if np.max(np.abs(v_new - v)) < RVI_TOL:
                break
            v = v_new

        # Extract optimal policy
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            vals = l_sa[s] + self.P[s] @ v
            policy[s] = np.argmin(vals)

        # Build transition matrix for the policy
        Ppi = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            Ppi[s] = self.P[s, policy[s]]

        # Compute stationary distribution
        pi = self.stationary_distribution(Ppi)

        # Calculate performance metrics
        RE = sum(pi[s] * self.c_sa[s, policy[s]] for s in range(self.num_states))
        F1 = sum(pi[s] * self.f1_a[policy[s]] for s in range(self.num_states))
        F2 = sum(pi[s] * self.f2_a[policy[s]] for s in range(self.num_states))

        return policy, RE, F1, F2

    # ----------------- Inner Lambda2 Bisection Search (with lambda2 mixing) -----------------
    def inner_lambda2_search(self, lambda1, F2_max):
        """
        Inner bisection search: fix lambda1, search lambda2,
        and mix policies at lambda2^- and lambda2^+ to meet F2_max exactly
        """
        lam2_low = 0.0
        lam2_high = LAMBDA_MAX

        # Evaluate policies at lambda2 bounds
        pol_low, RE_low, F1_low, F2_low = self.RVI(lambda1, lam2_low)
        pol_high, RE_high, F1_high, F2_high = self.RVI(lambda1, lam2_high)

        # Bisection search
        for _ in range(MAX_INNER):
            lam2_mid = 0.5 * (lam2_low + lam2_high)
            pol_mid, RE_mid, F1_mid, F2_mid = self.RVI(lambda1, lam2_mid)

            if F2_mid > F2_max:
                # F2 too high, need larger lambda2
                lam2_low = lam2_mid
                pol_low, RE_low, F1_low, F2_low = pol_mid, RE_mid, F1_mid, F2_mid
            else:
                # F2 within constraint, try smaller lambda2
                lam2_high = lam2_mid
                pol_high, RE_high, F1_high, F2_high = pol_mid, RE_mid, F1_mid, F2_mid

            if abs(lam2_high - lam2_low) < BISEC_TOL:
                break

        # Mix policies at lambda2 bounds to hit F2_max exactly
        denom = F2_high - F2_low
        if abs(denom) < 1e-12:
            mu = 0.5
        else:
            mu = (F2_high - F2_max) / denom
        mu = float(np.clip(mu, 0.0, 1.0))  # Ensure mixing coefficient is valid

        # Calculate mixed performance
        RE_mix = mu * RE_low + (1 - mu) * RE_high
        F1_mix = mu * F1_low + (1 - mu) * F1_high
        F2_mix = mu * F2_low + (1 - mu) * F2_high

        # Build mixed stochastic policy pi_λ2(s,a)
        S, A = self.num_states, self.num_actions
        pi_mix_lambda2 = np.zeros((S, A))
        for s in range(S):
            pi_mix_lambda2[s, pol_low[s]] += mu
            pi_mix_lambda2[s, pol_high[s]] += (1 - mu)

        return {
            "RE": RE_mix,
            "F1": F1_mix,
            "F2": F2_mix,
            "lam2_low": lam2_low,
            "lam2_high": lam2_high,
            "pol_low": pol_low,
            "pol_high": pol_high,
            "mu": mu,
            "pi_mix": pi_mix_lambda2
        }

    # ----------------- Outer Lambda1 Bisection Search (with mixing) -----------------
    def outer_lambda1_search(self, F1_max, F2_max):
        """
        Outer bisection search: call inner_lambda2_search at lambda1^- and lambda1^+,
        then mix the resulting policies to meet F1_max exactly
        """

        # 1) First check lambda1 = 0 case
        res_low = self.inner_lambda2_search(0.0, F2_max)
        F1_low = res_low["F1"]

        # If lambda1=0 already satisfies F1 constraint (and F2 is already at F2_max),
        # constraint is inactive - return this solution directly
        if F1_low <= F1_max + BISEC_TOL:
            return {
                "RE": res_low["RE"],
                "F1": res_low["F1"],
                "F2": res_low["F2"],
                "lam1": 0.0,
                "lam2_low": res_low["lam2_low"],
                "lam2_high": res_low["lam2_high"],
                "mu_inner": res_low["mu"],
                "nu_outer": 1.0,  # Only use low policy
                "pi_mix": res_low["pi_mix"]
            }

        # 2) Find lambda1_high such that F1_high <= F1_max
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

        # If even at lambda1_high, F1_high > F1_max (very tight constraint or numerical issue)
        # Return the solution at lambda1_high (minimum achievable F1)
        if F1_high > F1_max + BISEC_TOL:
            return {
                "RE": res_high["RE"],
                "F1": res_high["F1"],
                "F2": res_high["F2"],
                "lam1": lam1_high,
                "lam2_low": res_high["lam2_low"],
                "lam2_high": res_high["lam2_high"],
                "mu_inner": res_high["mu"],
                "nu_outer": 1.0,
                "pi_mix": res_high["pi_mix"]
            }

        # Now F1_low > F1_max >= F1_high: (lambda1_low, lambda1_high) crosses F1_max
        # 3) Formal outer bisection search for lambda1
        for _ in range(MAX_OUTER):
            lam1_mid = 0.5 * (lam1_low + lam1_high)
            res_mid = self.inner_lambda2_search(lam1_mid, F2_max)
            F1_mid = res_mid["F1"]

            if F1_mid > F1_max:  # Frequency still too high, need larger lambda1
                lam1_low = lam1_mid
                res_low = res_mid
            else:  # F1_mid <= F1_max
                lam1_high = lam1_mid
                res_high = res_mid

            if abs(lam1_high - lam1_low) < BISEC_TOL:
                break

        # 4) Mix the two "inner-mixed policies" at lambda1 bounds to hit F1_max exactly
        C_low, F1_low, F2_low, pi_low = res_low["RE"], res_low["F1"], res_low["F2"], res_low["pi_mix"]
        C_high, F1_high, F2_high, pi_high = res_high["RE"], res_high["F1"], res_high["F2"], res_high["pi_mix"]

        denom = F1_low - F1_high
        if abs(denom) < 1e-12:
            nu = 0.5
        else:
            # Solve nu·F1_low + (1-nu)·F1_high = F1_max
            nu = (F1_max - F1_high) / denom
        nu = float(np.clip(nu, 0.0, 1.0))

        # Calculate final mixed policy and performance
        RE_final = nu * C_low + (1 - nu) * C_high
        F1_final = nu * F1_low + (1 - nu) * F1_high
        F2_final = nu * F2_low + (1 - nu) * F2_high
        pi_final = nu * pi_low + (1 - nu) * pi_high

        # Take "expected value" of lambda1 (for reference only)
        lam1_star = nu * lam1_low + (1 - nu) * lam1_high

        return {
            "RE": RE_final,
            "F1": F1_final,
            "F2": F2_final,
            "lam1": lam1_star,
            "lam2_low": (res_low["lam2_low"], res_high["lam2_low"]),
            "lam2_high": (res_low["lam2_high"], res_high["lam2_high"]),
            "mu_inner": (res_low["mu"], res_high["mu"]),
            "nu_outer": nu,
            "pi_mix": pi_final
        }

    # ----------------- Double Bisection Search Main Entry -----------------
    def double_bisection_rvi(self):
        """Double bisection search with RVI to solve CMDP (inner lambda2 + outer lambda1 mixing)"""
        start_time = time.time()
        res = self.outer_lambda1_search(self.Fmax[0], self.Fmax[1])
        elapsed_time = time.time() - start_time

        res["compute_time"] = elapsed_time
        # Compatibility with field names used in main()
        return {
            'RE': res["RE"],
            'F1': res["F1"],
            'F2': res["F2"],
            'lambda1': res["lam1"],
            'lambda2_range': res["lam2_low"],  # Placeholder info, not critical
            'mix_coeff': {'mu_inner': res["mu_inner"], 'nu_outer': res["nu_outer"]},
            'mix_policy': res["pi_mix"],
            'compute_time': elapsed_time
        }

    # ----------------- Supplementary: simulate_transmission method (compatible with original simulation) -----------------
    def simulate_transmission(self, s_idx, action):
        """Simulate one transmission step for a given state and action"""
        a_idx = self.action_index[action]
        next_s_idx = np.random.choice(self.num_states, p=self.P[s_idx, a_idx])
        actual_re = self.c_sa[s_idx, a_idx]
        return next_s_idx, actual_re


# ============================================================
# 2. Utility Functions: Stationary Distribution, Policy Evaluation, Multi-run Simulation Averaging
# ============================================================
def stationary(Ppi, tol=1e-12, max_iter=10000):
    """Compute stationary distribution for transition matrix using power iteration"""
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
    """Theoretically evaluate policy performance using stationary distribution"""
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
    """Single simulation run (with independent seed)"""
    if seed is not None:
        np.random.seed(seed)

    S, A = cmdp.num_states, cmdp.num_actions

    # Initial state selection
    if mu0 is None:
        s = 0
    else:
        s = np.random.choice(S, p=mu0)

    C_sum = 0.0
    F1_sum = 0.0
    F2_sum = 0.0
    count = 0

    # Simulation loop
    for t in range(SIMULATION_STEPS + SIMULATION_BURN_IN):
        # Select action according to policy
        a = np.random.choice(A, p=pi[s])
        a1, a2 = cmdp.actions[a]

        # Collect statistics after burn-in period
        if t >= SIMULATION_BURN_IN:
            C_sum += cmdp.c_sa[s, a]
            F1_sum += (1 if a1 != 0 else 0)
            F2_sum += (1 if a2 != 0 else 0)
            count += 1

        # Transition to next state
        s = np.random.choice(S, p=cmdp.P[s, a])

    # Return average values
    return C_sum / count, F1_sum / count, F2_sum / count


def simulate_policy_multiple(cmdp: TwoSourceMPRCMDP, pi, mu0=None, times=SIMULATION_TIMES):
    """Multiple simulation runs with averaging, return (mean, standard deviation)"""
    c_list, f1_list, f2_list = [], [], []

    # Use different random seeds for each run to ensure independence
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
# 3. Occupation-Measure LP for CMDP Solution
# ============================================================
def solve_cmdp_via_lp(cmdp: TwoSourceMPRCMDP):
    """Solve CMDP using occupation-measure linear programming"""
    S, A = cmdp.num_states, cmdp.num_actions
    N = S * A  # Total number of decision variables (occupation measures)

    # Helper function for variable indexing
    def idx(s, a):
        return s * A + a

    # Build cost and constraint vectors
    c_vec = np.zeros(N)       # Objective: minimize expected RE
    F1_vec = np.zeros(N)      # Frequency constraint for source 1
    F2_vec = np.zeros(N)      # Frequency constraint for source 2

    for s in range(S):
        for a in range(A):
            i = idx(s, a)
            c_vec[i] = cmdp.c_sa[s, a]
            F1_vec[i] = cmdp.f1_a[a]
            F2_vec[i] = cmdp.f2_a[a]

    # Build equality constraints (flow conservation + normalization)
    A_eq = np.zeros((S + 1, N))
    b_eq = np.zeros(S + 1)

    # Flow conservation constraints: sum_a x(s,a) = sum_{s',a'} x(s',a')P(s',a',s)
    for sp in range(S):
        for s in range(S):
            for a in range(A):
                A_eq[sp, idx(s, a)] += (1.0 if s == sp else 0.0) - cmdp.P[s, a, sp]
        b_eq[sp] = 0.0

    # Normalization constraint: sum_{s,a} x(s,a) = 1
    A_eq[S, :] = 1.0
    b_eq[S] = 1.0

    # Build inequality constraints (frequency constraints)
    A_ub = np.vstack([F1_vec, F2_vec])
    b_ub = cmdp.Fmax.copy()

    # Variable bounds (occupation measures are non-negative)
    bounds = [(0.0, None) for _ in range(N)]

    # Solve LP using HiGHS solver
    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError("LP solve failed: " + res.message)

    # Reshape solution to state-action matrix
    x = res.x.reshape(S, A)

    # Compute stationary state distribution
    mu_s = x.sum(axis=1)

    # Convert occupation measures to stochastic policy
    pi = np.zeros((S, A))
    for s in range(S):
        if mu_s[s] > 1e-12:
            pi[s, :] = x[s, :] / mu_s[s]
        else:
            pi[s, :] = 1.0 / A  # Uniform random policy for unreachable states

    # Calculate optimal performance
    C_opt = float((cmdp.c_sa * x).sum())
    F1_opt = float((cmdp.f1_a[np.newaxis, :] * x).sum())
    F2_opt = float((cmdp.f2_a[np.newaxis, :] * x).sum())

    return x, pi, mu_s, C_opt, F1_opt, F2_opt


# ============================================================
# 4. Fixed Probability Strategy
# ============================================================
def build_fixed_policy(cmdp: TwoSourceMPRCMDP):
    """Build fixed-probability transmission policy"""
    A = cmdp.num_actions
    S = cmdp.num_states
    F1_max, F2_max = cmdp.Fmax

    # Fixed transmission probabilities for each source
    p1 = {0: 1 - F1_max, 1: F1_max / 2, 2: F1_max / 2}  # Source 1 transmission prob
    p2 = {0: 1 - F2_max, 1: F2_max / 2, 2: F2_max / 2}  # Source 2 transmission prob

    # Joint action probabilities (independent)
    pi_a = np.zeros(A, dtype=float)
    for idx, (a1, a2) in enumerate(cmdp.actions):
        pi_a[idx] = p1[a1] * p2[a2]

    # Same policy for all states
    pi = np.tile(pi_a, (S, 1))
    return pi


# ============================================================
# 5. DPP (Drift Plus Penalty) Algorithm Class
# ============================================================
class LowComplexityDPP:
    """Low-complexity Drift Plus Penalty (DPP) algorithm for CMDP"""
    def __init__(self, cmdp, V1=100, V2=100):
        self.cmdp = cmdp
        self.V = np.array([V1, V2])  # DPP weight parameters
        self.Z = np.zeros(2)         # Virtual queues for constraints
        self.reset()

    def reset(self):
        """Reset algorithm state"""
        self.Z = np.zeros(2)
        self.total_re = 0.0
        self.transmission_counts = np.zeros(2)
        self.time_slot = 0
        self.current_state_idx = 0
        self.current_state = self.cmdp.states[0]

    def compute_dpp_objective(self, state_idx, action_idx):
        """Compute DPP objective function for state-action pair"""
        expected_re = self.cmdp.c_sa[state_idx, action_idx]
        f1 = self.cmdp.f1_a[action_idx]
        f2 = self.cmdp.f2_a[action_idx]
        # DPP objective: minimize V*RE + Z1*f1 + Z2*f2
        return self.V[0] * expected_re + self.Z[0] * f1 + self.Z[1] * f2

    def select_action(self):
        """Select optimal action using DPP objective"""
        best_value = float('inf')
        best_action_idx = 0
        for a_idx in range(self.cmdp.num_actions):
            val = self.compute_dpp_objective(self.current_state_idx, a_idx)
            if val < best_value:
                best_value = val
                best_action_idx = a_idx
        return best_action_idx, self.cmdp.actions[best_action_idx]

    def update_virtual_queues(self, action):
        """Update virtual queues based on selected action"""
        a1, a2 = action
        f1 = 1 if a1 != 0 else 0
        f2 = 1 if a2 != 0 else 0
        # Queue update: Z_i = max(Z_i - Fmax_i + f_i, 0)
        self.Z[0] = max(self.Z[0] - self.cmdp.Fmax[0] + f1, 0)
        self.Z[1] = max(self.Z[1] - self.cmdp.Fmax[1] + f2, 0)

    def simulate_transmission(self, action):
        """Simulate one transmission step and update state"""
        a1, a2 = action
        s_idx = self.current_state_idx
        a_idx = self.cmdp.action_index[action]
        next_state_idx = np.random.choice(self.cmdp.num_states, p=self.cmdp.P[s_idx, a_idx])
        actual_re = self.cmdp.c_sa[s_idx, a_idx]
        self.current_state_idx = next_state_idx
        self.current_state = self.cmdp.states[next_state_idx]
        return next_state_idx, actual_re

    def run_simulation(self, num_steps=50000, burn_in=2000, count_actions=False):
        """Run DPP simulation and collect performance statistics"""
        self.reset()
        self.stats = {
            "single1": 0, "single2": 0, "both": 0, "none": 0,
            "s1_src1": 0, "s1_src2": 0, "s2_src1": 0, "s2_src2": 0
        }

        # Burn-in period
        for _ in range(burn_in):
            a_idx, action = self.select_action()
            self.simulate_transmission(action)
            self.update_virtual_queues(action)

        # Main simulation
        for _ in range(num_steps):
            a_idx, action = self.select_action()
            next_state_idx, re = self.simulate_transmission(action)
            self.update_virtual_queues(action)

            a1, a2 = action
            f1 = 1 if a1 != 0 else 0
            f2 = 1 if a2 != 0 else 0

            # Accumulate statistics
            self.total_re += re
            self.transmission_counts[0] += f1
            self.transmission_counts[1] += f2

            # Detailed action statistics (optional)
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

        # Normalize results
        avg_re = self.total_re / num_steps
        avg_freq = self.transmission_counts / num_steps
        if count_actions:
            for k in self.stats:
                self.stats[k] /= num_steps

        return avg_re, avg_freq


# ---------------- Enhanced DPP (Complete Statistics) ----------------
class EnhancedDPP(LowComplexityDPP):
    """Enhanced DPP with detailed action statistics collection"""
    def run_simulation(self, num_steps=50000, burn_in=2000, count_actions=True):
        return super().run_simulation(num_steps, burn_in, count_actions)


# ============================================================
# 6. Enhanced Simulation Function (Statistics for Fast/Slow Source Scheduling)
# ============================================================
def simulate_policy_mc_enhanced(cmdp, policy, num_steps=50000, burn_in=2000):
    """Monte-Carlo simulation (enhanced): statistics for fast/slow source scheduling and transmission modes"""
    S, A = cmdp.num_states, cmdp.num_actions
    state = 0
    total_re = 0.0
    total_f1 = 0
    total_f2 = 0
    stats = {
        "single1": 0, "single2": 0, "both": 0, "none": 0,
        "s1_src1": 0, "s1_src2": 0, "s2_src1": 0, "s2_src2": 0
    }

    # Burn-in period
    for _ in range(burn_in):
        a_idx = np.random.choice(A, p=policy[state])
        state = np.random.choice(S, p=cmdp.P[state, a_idx])

    # Main simulation
    for _ in range(num_steps):
        a_idx = np.random.choice(A, p=policy[state])
        a1, a2 = cmdp.actions[a_idx]
        next_state = np.random.choice(S, p=cmdp.P[state, a_idx])

        # Accumulate RE and frequency statistics
        total_re += cmdp.c_sa[state, a_idx]
        f1 = 1 if a1 != 0 else 0
        f2 = 1 if a2 != 0 else 0
        total_f1 += f1
        total_f2 += f2

        # Action mode statistics
        if f1 and not f2:
            stats["single1"] += 1
        elif f2 and not f1:
            stats["single2"] += 1
        elif f1 and f2:
            stats["both"] += 1
        else:
            stats["none"] += 1

        # Source type statistics
        if a1 == 1:
            stats["s1_src1"] += 1  # Sensor 1 transmits slow source
        elif a1 == 2:
            stats["s1_src2"] += 1  # Sensor 1 transmits fast source
        if a2 == 1:
            stats["s2_src1"] += 1  # Sensor 2 transmits slow source
        elif a2 == 2:
            stats["s2_src2"] += 1  # Sensor 2 transmits fast source

        state = next_state

    # Normalize results
    cnt = num_steps
    result = {
        "avg_re": total_re / cnt,
        "freq1": total_f1 / cnt,
        "freq2": total_f2 / cnt
    }
    for k in stats:
        result[k] = stats[k] / cnt
    return result


# ============================================================
# 7. Main Evaluation Function: Analyze Fast/Slow Source Dynamics
# ============================================================
def evaluate_source_dynamics():
    """Evaluate fast/slow source scheduling frequency, transmission modes, actual frequency & RE"""
    results = []
    # Fixed source transition probabilities (fast source Q1, slow source Q2)
    Q1_fast = np.array([[0.8, 0.2],
                        [0.2, 0.8]])  # Slow source: low transition probability
    Q2_slow = np.array([[0.2, 0.8],
                        [0.8, 0.2]])  # Fast source: high transition probability

    print(f"\n=== Evaluation Configuration ===")
    print(f"Fixed Fmax: {FMAX_FIXED} | Slow source Q1: {Q1_fast} | Fast source Q2: {Q2_slow}")
    print(f"Simulation steps: {SIMULATION_STEPS} | Burn-in steps: {SIMULATION_BURN_IN}")
    print(f"Channel success probability range: {P_RANGE}\n")

    for p in P_RANGE:
        print(f">>> Channel success probability p = {p:.1f}")
        row = {"p": float(p)}

        # Build CMDP (fixed fast/slow sources, sweep channel success probability)
        cmdp = TwoSourceMPRCMDP(
            Q1=Q1_fast, Q2=Q2_slow,
            p1_only=p, p2_only=p, p1_both=p / 2, p2_both=p / 2,
            Fmax1=FMAX_FIXED, Fmax2=FMAX_FIXED
        )

        # 1. LP algorithm
        try:
            _, pi_lp, _, _, _, _ = solve_cmdp_via_lp(cmdp)
            row["LP"] = simulate_policy_mc_enhanced(cmdp, pi_lp, SIMULATION_STEPS, SIMULATION_BURN_IN)
            print(f"  LP: RE={row['LP']['avg_re']:.4f}, F1={row['LP']['freq1']:.4f}, F2={row['LP']['freq2']:.4f}")
        except Exception as e:
            row["LP"] = None
            print(f"  LP: Failed - {e}")

        # 2. RVI algorithm
        try:
            rvi_res = cmdp.double_bisection_rvi()
            pi_rvi = rvi_res["mix_policy"]
            row["RVI"] = simulate_policy_mc_enhanced(cmdp, pi_rvi, SIMULATION_STEPS, SIMULATION_BURN_IN)
            print(f"  RVI: RE={row['RVI']['avg_re']:.4f}, F1={row['RVI']['freq1']:.4f}, F2={row['RVI']['freq2']:.4f}")
        except Exception as e:
            row["RVI"] = None
            print(f"  RVI: Failed - {e}")

        # 3. FPS (Fixed Probability Strategy)
        try:
            pi_fps = build_fixed_policy(cmdp)
            row["FPS"] = simulate_policy_mc_enhanced(cmdp, pi_fps, SIMULATION_STEPS, SIMULATION_BURN_IN)
            print(f"  FPS: RE={row['FPS']['avg_re']:.4f}, F1={row['FPS']['freq1']:.4f}, F2={row['FPS']['freq2']:.4f}")
        except Exception as e:
            row["FPS"] = None
            print(f"  FPS: Failed - {e}")

        # 4. DPP algorithm
        try:
            dpp = EnhancedDPP(cmdp, V1=V_DPP, V2=V_DPP)
            avg_re, (freq1, freq2) = dpp.run_simulation(SIMULATION_STEPS, SIMULATION_BURN_IN)
            row["DPP"] = {
                "avg_re": avg_re,
                "freq1": freq1,
                "freq2": freq2,
                "single1": dpp.stats["single1"],
                "single2": dpp.stats["single2"],
                "both": dpp.stats["both"],
                "none": dpp.stats["none"],
                "s1_src1": dpp.stats["s1_src1"],  # Sensor 1 transmits slow source
                "s1_src2": dpp.stats["s1_src2"],  # Sensor 1 transmits fast source
                "s2_src1": dpp.stats["s2_src1"],  # Sensor 2 transmits slow source
                "s2_src2": dpp.stats["s2_src2"]   # Sensor 2 transmits fast source
            }
            print(f"  DPP: RE={avg_re:.4f}, F1={freq1:.4f}, F2={freq2:.4f}")
        except Exception as e:
            row["DPP"] = None
            print(f"  DPP: Failed - {e}")

        results.append(row)
        print("-" * 100)

    return results


# ============================================================
# 9. Result Saving & Summary Output
# ============================================================
def save_results_flat(results):
    """Save results in flat format for easy analysis"""

    p_vals = np.array([r["p"] for r in results])
    algorithms = ["LP", "RVI", "FPS", "DPP"]
    metrics = ["avg_re", "freq1", "freq2", "single1", "single2",
               "both", "none", "s1_src1", "s1_src2", "s2_src1", "s2_src2"]

    # Initialize empty arrays: one array per algorithm×metric
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

    # Save as MAT file
    try:
        import scipy.io as sio
        sio.savemat("result/REvsP.mat", flat_data)
        print("\nResults saved to REvsP.mat")
    except:
        print("\nscipy.io not found, MAT file not saved")

    # Print variable names (optional)
    print("\nSaved variable names:")
    for k in flat_data:
        print("  ", k)

    return flat_data


# ============================================================
# 10. New: Print Multi-dimensional Performance Summary Table
# ============================================================
def print_performance_summary(results):
    """Print detailed performance summary table with all core metrics"""
    algorithms = ["LP", "RVI", "FPS", "DPP"]
    metrics = [
        ("avg_re", "Average RE"),
        ("freq1", "Sensor 1 Freq"),
        ("freq2", "Sensor 2 Freq"),
        ("single1", "Only Sensor 1 Tx"),
        ("single2", "Only Sensor 2 Tx"),
        ("both", "Both Sensors Tx"),
        ("s1_src1", "Sensor1-SlowSrc"),
        ("s1_src2", "Sensor1-FastSrc"),
        ("s2_src1", "Sensor2-SlowSrc"),
        ("s2_src2", "Sensor2-FastSrc")
    ]

    # Print header
    print("\n" + "=" * 180)
    print("Multi-dimensional Performance Summary (Grouped by Channel Success Probability p)")
    print("=" * 180)

    # Iterate over each p value and print metrics for all algorithms
    for row in results:
        p = row["p"]
        print(f"\n--- Channel Success Probability p = {p:.1f} ---")

        # Build table header
        table_header = f"{'Algo':<8}"
        for _, metric_name in metrics:
            table_header += f"| {metric_name:<12}"
        print(table_header)
        print("-" * 180)

        # Print metrics for each algorithm
        for algo in algorithms:
            if row[algo] is None:
                # Algorithm failed
                row_str = f"{algo:<8}"
                for _ in metrics:
                    row_str += f"| {'Failed':<12}"
                print(row_str)
            else:
                # Algorithm succeeded, print values
                row_str = f"{algo:<8}"
                for metric_key, _ in metrics:
                    val = row[algo][metric_key]
                    row_str += f"| {val:<12.4f}"
                print(row_str)

    # Print average summary (by algorithm)
    print("\n" + "=" * 180)
    print("Average Performance Summary (Averaged over all p values)")
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
    avg_header = f"{'Algo':<8}"
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

    # Print optimal algorithm marker (minimum RE)
    print("\n" + "=" * 180)
    print("Optimal Algorithm by RE (Smaller RE is Better)")
    print("=" * 180)
    print(f"{'p value':<8} | {'Best Algo':<8} | {'Min RE':<12} | {'2nd Best':<8} | {'2nd RE':<12}")
    print("-" * 180)

    for row in results:
        p = row["p"]
        re_vals = {}
        for algo in algorithms:
            if row[algo] is not None:
                re_vals[algo] = row[algo]["avg_re"]

        if re_vals:
            # Sort RE values
            sorted_re = sorted(re_vals.items(), key=lambda x: x[1])
            best_algo, best_re = sorted_re[0]
            if len(sorted_re) >= 2:
                second_algo, second_re = sorted_re[1]
            else:
                second_algo, second_re = "-", "-"

            print(f"{p:.1f}      | {best_algo:<8} | {best_re:<12.4f} | {second_algo:<8} | {second_re:<12.4f}")
        else:
            print(f"{p:.1f}      | {'No Valid':<8} | {'-':<12} | {'-':<8} | {'-':<12}")


# ============================================================
# 11. Original Main Function (Preserved for Separate Execution)
# ============================================================
def main_original():
    p_list = np.arange(0.1, 1.01, 0.1)
    F1_max = 0.4
    F2_max = 0.4

    # Storage for means and standard deviations
    lp_C_th, lp_C_mc_mean, lp_C_mc_std = [], [], []
    bisection_C_th, bisection_C_mc_mean, bisection_C_mc_std = [], [], []
    fix_C_th, fix_C_mc_mean, fix_C_mc_std = [], [], []

    lp_F1_mc_mean, lp_F1_mc_std = [], []
    bisection_F1_mc_mean, bisection_F1_mc_std = [], []
    fix_F1_mc_mean, fix_F1_mc_std = [], []

    lp_F2_mc_mean, lp_F2_mc_std = [], []
    bisection_F2_mc_mean, bisection_F2_mc_std = [], []
    fix_F2_mc_mean, fix_F2_mc_std = [], []

    print(f"===== Simulation Configuration =====")
    print(f"Simulation runs per policy: {SIMULATION_TIMES}")
    print(f"Burn-in steps per simulation: {SIMULATION_BURN_IN}")
    print(f"Effective steps per simulation: {SIMULATION_STEPS}")
    print(f"====================================\n")

    for p in p_list:
        cmdp = TwoSourceMPRCMDP(
            p1_only=p,
            p2_only=p,
            p1_both=p / 2,
            p2_both=p / 2,
            Fmax1=F1_max,
            Fmax2=F2_max,

        )

        print(f"\n===== p = {p:.1f} =====")

        # ----- 1) LP Optimal -----
        try:
            X, pi_lp, mu_lp, C_lp, F1_lp, F2_lp = solve_cmdp_via_lp(cmdp)
            # Multiple simulations for averaging
            (c_mean, c_std), (f1_mean, f1_std), (f2_mean, f2_std) = simulate_policy_multiple(cmdp, pi_lp, mu0=mu_lp)
        except Exception as e:
            print(f"LP solution failed: {e}")
            C_lp = float('inf')
            c_mean = c_std = f1_mean = f1_std = f2_mean = f2_std = float('inf')

        lp_C_th.append(C_lp)
        lp_C_mc_mean.append(c_mean)
        lp_C_mc_std.append(c_std)
        lp_F1_mc_mean.append(f1_mean)
        lp_F1_mc_std.append(f1_std)
        lp_F2_mc_mean.append(f2_mean)
        lp_F2_mc_std.append(f2_std)

        # ----- 2) Double Bisection RVI -----
        try:
            bisection_res = cmdp.double_bisection_rvi()
            C_bisection = bisection_res['RE']
            F1_bisection = bisection_res['F1']
            F2_bisection = bisection_res['F2']

            mu_bis, C_bisection_th, F1_bisection_th, F2_bisection_th = eval_policy_theoretical(
                cmdp, bisection_res['mix_policy']
            )
            # Multiple simulations for averaging
            (c_mean, c_std), (f1_mean, f1_std), (f2_mean, f2_std) = simulate_policy_multiple(cmdp, bisection_res[
                'mix_policy'], mu0=mu_bis)
        except Exception as e:
            print(f"Double bisection RVI solution failed: {e}")
            C_bisection_th = float('inf')
            c_mean = c_std = f1_mean = f1_std = f2_mean = f2_std = float('inf')

        bisection_C_th.append(C_bisection_th)
        bisection_C_mc_mean.append(c_mean)
        bisection_C_mc_std.append(c_std)
        bisection_F1_mc_mean.append(f1_mean)
        bisection_F1_mc_std.append(f1_std)
        bisection_F2_mc_mean.append(f2_mean)
        bisection_F2_mc_std.append(f2_std)

        # ----- 3) Fixed Strategy -----
        try:
            pi_fix = build_fixed_policy(cmdp)
            mu_fix, C_fix_th, F1_fix_th, F2_fix_th = eval_policy_theoretical(cmdp, pi_fix)
            # Multiple simulations for averaging
            (c_mean, c_std), (f1_mean, f1_std), (f2_mean, f2_std) = simulate_policy_multiple(cmdp, pi_fix, mu0=mu_fix)
        except Exception as e:
            print(f"Fixed strategy solution failed: {e}")
            C_fix_th = float('inf')
            c_mean = c_std = f1_mean = f1_std = f2_mean = f2_std = float('inf')

        fix_C_th.append(C_fix_th)
        fix_C_mc_mean.append(c_mean)
        fix_C_mc_std.append(c_std)
        fix_F1_mc_mean.append(f1_mean)
        fix_F1_mc_std.append(f1_std)
        fix_F2_mc_mean.append(f2_mean)
        fix_F2_mc_std.append(f2_std)

        # Print results for current p value (with standard deviation)
        print(f"[LP]      C={lp_C_th[-1]:.4f} (MC mean={lp_C_mc_mean[-1]:.4f}, std={lp_C_mc_std[-1]:.4f}), "
              f"F1(MC)={lp_F1_mc_mean[-1]:.4f}±{lp_F1_mc_std[-1]:.4f}, "
              f"F2(MC)={lp_F2_mc_mean[-1]:.4f}±{lp_F2_mc_std[-1]:.4f}")

        print(
            f"[Double-Bisection RVI] C={bisection_C_th[-1]:.4f} (MC mean={bisection_C_mc_mean[-1]:.4f}, std={bisection_C_mc_std[-1]:.4f}), "
            f"F1(MC)={bisection_F1_mc_mean[-1]:.4f}±{bisection_F1_mc_std[-1]:.4f}, "
            f"F2(MC)={bisection_F2_mc_mean[-1]:.4f}±{bisection_F2_mc_std[-1]:.4f}")

        print(f"[Fixed]   C={fix_C_th[-1]:.4f} (MC mean={fix_C_mc_mean[-1]:.4f}, std={fix_C_mc_std[-1]:.4f}), "
              f"F1(MC)={fix_F1_mc_mean[-1]:.4f}±{fix_F1_mc_std[-1]:.4f}, "
              f"F2(MC)={fix_F2_mc_mean[-1]:.4f}±{fix_F2_mc_std[-1]:.4f}")

    # ========================================================
    # Plot: Average RE (with error bars)
    # ========================================================
    p_vals = p_list

    # -------- Summary Table --------
    print("\n" + "=" * 120)
    print("Performance Summary (Theoretical + Monte Carlo Mean)")
    print("=" * 120)
    print(
        f"{'p_only':<8} {'LP RE':<10} {'RVI RE':<15} {'Fixed RE':<15} {'RVI-LP Gap(%)':<12} {'Fixed-LP Gap(%)':<12} {'LP RE MC Mean':<15} {'RVI RE MC Mean':<15} {'Fixed RE MC Mean':<15}")
    print("-" * 120)
    for i, p in enumerate(p_vals):
        if lp_C_th[i] == float('inf'):
            continue
        rvi_gap = (bisection_C_th[i] - lp_C_th[i]) / lp_C_th[i] * 100 if lp_C_th[i] != 0 else 0
        fix_gap = (fix_C_th[i] - lp_C_th[i]) / lp_C_th[i] * 100 if lp_C_th[i] != 0 else 0
        print(
            f"{p:<8.1f} {lp_C_th[i]:<10.4f} {bisection_C_th[i]:<15.4f} {fix_C_th[i]:<15.4f} {rvi_gap:<12.2f} {fix_gap:<12.2f} {lp_C_mc_mean[i]:<15.4f} {bisection_C_mc_mean[i]:<15.4f} {fix_C_mc_mean[i]:<15.4f}")


# ============================================================
# Main Entry: Run Fast/Slow Source Analysis + Print Summary Table
# ============================================================
if __name__ == "__main__":
    # Run fast/slow source scheduling analysis (core functionality)
    results = evaluate_source_dynamics()

    # Print detailed performance summary table (new)
    print_performance_summary(results)

    # Save results & output summary
    flat_data = save_results_flat(results)