import numpy as np
from itertools import product
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import time
from CAEvsP_compare import *


# ============================================================
# 低复杂度DPP算法类
# ============================================================
class LowComplexityDPP:
    """低复杂度在线DPP算法"""

    def __init__(self, cmdp, V1=100.0, V2=100.0):
        """
        初始化DPP算法

        参数:
        - cmdp: TwoSourceMPRCMDP对象
        - V1, V2: 权重参数，平衡CAE和队列稳定性
        """
        self.cmdp = cmdp
        self.V = np.array([V1, V2])

        # 初始化虚拟队列
        self.Z = np.zeros(2)

        # 传输频率约束
        self.Fmax = cmdp.Fmax.copy()

        # 统计信息
        self.total_cae = 0.0
        self.transmission_counts = np.zeros(2)
        self.time_slot = 0

        # 状态跟踪
        self.current_state_idx = 0
        self.current_state = self.cmdp.states[self.current_state_idx]

        # 动作空间大小
        self.num_actions = cmdp.num_actions

        # 缓存每个状态-动作对的期望CAE
        self.c_sa = cmdp.c_sa.copy()

    def reset(self):
        """重置算法状态"""
        self.Z = np.zeros(2)
        self.total_cae = 0.0
        self.transmission_counts = np.zeros(2)
        self.time_slot = 0
        self.current_state_idx = 0
        self.current_state = self.cmdp.states[0]

    def compute_dpp_objective(self, state_idx, action_idx):
        """
        计算DPP目标函数值

        目标: V * E[CAE] + Z1 * f1 + Z2 * f2
        """
        # 期望CAE
        expected_cae = self.c_sa[state_idx, action_idx]

        # 传输指示函数
        f1 = self.cmdp.f1_a[action_idx]
        f2 = self.cmdp.f2_a[action_idx]

        # DPP目标值
        dpp_value = (self.V[0] * expected_cae +
                     self.Z[0] * f1 +
                     self.Z[1] * f2)

        return dpp_value

    def select_action(self):
        """
        基于DPP算法选择最优动作

        复杂度: O(A) = O(9)，极低
        """
        best_value = float('inf')
        best_action_idx = 0

        # 遍历所有9个动作
        for a_idx in range(self.num_actions):
            dpp_value = self.compute_dpp_objective(self.current_state_idx, a_idx)

            if dpp_value < best_value:
                best_value = dpp_value
                best_action_idx = a_idx

        best_action = self.cmdp.actions[best_action_idx]
        return best_action_idx, best_action

    def update_virtual_queues(self, action):
        """
        更新虚拟队列

        更新规则: Z_i = max(Z_i - Fmax_i + f_i, 0)
        """
        a1, a2 = action
        f1 = 1 if a1 != 0 else 0
        f2 = 1 if a2 != 0 else 0

        # 更新队列
        self.Z[0] = max(self.Z[0] - self.Fmax[0] + f1, 0)
        self.Z[1] = max(self.Z[1] - self.Fmax[1] + f2, 0)

    def simulate_transmission(self, action):
        """
        模拟传输和状态转移

        返回:
        - next_state_idx: 下一状态索引
        - actual_cae: 实际CAE
        """
        a1, a2 = action
        current_state = self.current_state
        x1, x2, xhat1, xhat2 = current_state

        # 获取信道成功概率
        p_s1, p_s2 = self.cmdp.channel_success_probs(a1, a2)

        # 模拟信道成功
        s1_success = np.random.random() < p_s1 if p_s1 > 0 else False
        s2_success = np.random.random() < p_s2 if p_s2 > 0 else False

        # 确定哪个源被成功传输
        succ_src1 = (a1 == 1 and s1_success) or (a2 == 1 and s2_success)
        succ_src2 = (a1 == 2 and s1_success) or (a2 == 2 and s2_success)

        # 根据转移概率模拟源状态转移
        next_x1 = np.random.choice([0, 1], p=self.cmdp.Q1[x1])
        next_x2 = np.random.choice([0, 1], p=self.cmdp.Q2[x2])

        # 更新重构状态
        next_xhat1 = next_x1 if succ_src1 else xhat1
        next_xhat2 = next_x2 if succ_src2 else xhat2

        # 构建下一状态
        next_state = (next_x1, next_x2, next_xhat1, next_xhat2)
        next_state_idx = self.cmdp.state_index[next_state]

        # 计算实际CAE
        actual_cae = (self.cmdp.delta1[next_x1, next_xhat1] +
                      self.cmdp.delta2[next_x2, next_xhat2])

        return next_state_idx, actual_cae

    # def run_simulation(self, num_steps=50000, burn_in=2000):
    #     """
    #     运行DPP算法仿真
    #
    #     参数:
    #     - num_steps: 仿真时隙数
    #     - burn_in: 热身时隙数
    #
    #     返回:
    #     - avg_cae: 平均CAE
    #     - avg_freq: 平均传输频率
    #     """
    #     self.reset()
    #
    #     # 热身阶段
    #     for t in range(burn_in):
    #         action_idx, action = self.select_action()
    #         next_state_idx, actual_cae = self.simulate_transmission(action)
    #         self.update_virtual_queues(action)
    #
    #         # 更新状态
    #         self.current_state_idx = next_state_idx
    #         self.current_state = self.cmdp.states[next_state_idx]
    #
    #     # 主仿真阶段
    #     self.total_cae = 0.0
    #     self.transmission_counts = np.zeros(2)
    #     self.time_slot = 0
    #
    #     for t in range(num_steps):
    #         # 选择动作
    #         action_idx, action = self.select_action()
    #
    #         # 模拟传输
    #         next_state_idx, actual_cae = self.simulate_transmission(action)
    #
    #         # 更新虚拟队列
    #         self.update_virtual_queues(action)
    #
    #         # 更新统计
    #         self.total_cae += actual_cae
    #         self.transmission_counts[0] += 1 if action[0] != 0 else 0
    #         self.transmission_counts[1] += 1 if action[1] != 0 else 0
    #         self.time_slot += 1
    #
    #         # 更新状态
    #         self.current_state_idx = next_state_idx
    #         self.current_state = self.cmdp.states[next_state_idx]
    #
    #     # 计算平均性能
    #     avg_cae = self.total_cae / num_steps
    #     avg_freq = self.transmission_counts / num_steps
    #
    #     return avg_cae, avg_freq
    def run_simulation(self, num_steps=50000, burn_in=2000, count_actions=False):
        """
        运行 DPP 仿真并可统计动作情况。
        """

        # 重置
        self.reset()

        # 初始化动作统计
        self.stats = {
            "single1": 0,
            "single2": 0,
            "both": 0,
            "none": 0,
        }

        # =========================
        # 热身
        # =========================
        for _ in range(burn_in):
            a_idx, action = self.select_action()
            next_state_idx, cae = self.simulate_transmission(action)
            self.update_virtual_queues(action)

            self.current_state_idx = next_state_idx
            self.current_state = self.cmdp.states[next_state_idx]

        # =========================
        # 正式仿真
        # =========================
        self.total_cae = 0.0
        self.transmission_counts = np.zeros(2)

        for _ in range(num_steps):
            a_idx, action = self.select_action()

            next_state_idx, cae = self.simulate_transmission(action)
            self.update_virtual_queues(action)

            a1, a2 = action
            f1 = 1 if a1 != 0 else 0
            f2 = 1 if a2 != 0 else 0

            # 累积传输频率
            self.transmission_counts[0] += f1
            self.transmission_counts[1] += f2
            self.total_cae += cae

            # -------------------------
            # 动作统计
            # -------------------------
            if count_actions:
                if f1 == 1 and f2 == 0:
                    self.stats["single1"] += 1
                elif f1 == 0 and f2 == 1:
                    self.stats["single2"] += 1
                elif f1 == 1 and f2 == 1:
                    self.stats["both"] += 1
                else:
                    self.stats["none"] += 1

            self.current_state_idx = next_state_idx
            self.current_state = self.cmdp.states[next_state_idx]

        # =========================
        # 计算输出
        # =========================
        avg_cae = self.total_cae / num_steps
        avg_freq = self.transmission_counts / num_steps

        # 动作概率归一化
        if count_actions:
            for k in self.stats:
                self.stats[k] /= num_steps

        return avg_cae, avg_freq


class DPPSweepAnalyzer:
    """DPP算法参数扫描分析器"""

    @staticmethod
    def analyze_v_parameter(cmdp, V_values, num_steps=5000, num_runs=5):
        """
        分析V参数对DPP算法的影响

        返回:
        - results: 包含CAE和传输频率的字典
        """
        results = {
            'V': V_values,
            'avg_cae': [],
            'avg_freq1': [],
            'avg_freq2': [],
            'queue_variance': []
        }

        for V in V_values:
            print(f"  分析 V={V}...", end="")

            cae_list = []
            freq1_list = []
            freq2_list = []
            queue_var_list = []

            # 多次运行取平均
            for run in range(num_runs):
                dpp = LowComplexityDPP(cmdp, V1=V, V2=V)
                avg_cae, avg_freq = dpp.run_simulation(num_steps=num_steps)

                cae_list.append(avg_cae)
                freq1_list.append(avg_freq[0])
                freq2_list.append(avg_freq[1])
                queue_var_list.append(np.var(dpp.Z))

            results['avg_cae'].append(np.mean(cae_list))
            results['avg_freq1'].append(np.mean(freq1_list))
            results['avg_freq2'].append(np.mean(freq2_list))
            results['queue_variance'].append(np.mean(queue_var_list))

            print(f" CAE={results['avg_cae'][-1]:.4f}, F1={results['avg_freq1'][-1]:.4f}")

        return results


# ============================================================
# 修改主程序以包含DPP算法
# ============================================================
def main_with_dpp():
    p_list = np.arange(0.1, 1.01, 0.1)
    F1_max = 0.4
    F2_max = 0.4

    # 存储DPP算法的结果
    dpp_C_mc_mean, dpp_C_mc_std = [], []
    dpp_F1_mc_mean, dpp_F1_mc_std = [], []
    dpp_F2_mc_mean, dpp_F2_mc_std = [], []

    # 原有的结果存储（保持原样）
    lp_C_th, lp_C_mc_mean, lp_C_mc_std = [], [], []
    bisection_C_th, bisection_C_mc_mean, bisection_C_mc_std = [], [], []
    fix_C_th, fix_C_mc_mean, fix_C_mc_std = [], [], []

    lp_F1_mc_mean, lp_F1_mc_std = [], []
    bisection_F1_mc_mean, bisection_F1_mc_std = [], []
    fix_F1_mc_mean, fix_F1_mc_std = [], []

    lp_F2_mc_mean, lp_F2_mc_std = [], []
    bisection_F2_mc_mean, bisection_F2_mc_std = [], []
    fix_F2_mc_mean, fix_F2_mc_std = [], []

    print(f"===== 仿真配置 =====")
    print(f"每次策略仿真次数: {SIMULATION_TIMES}")
    print(f"单次仿真预热步数: {SIMULATION_BURN_IN}")
    print(f"单次仿真有效步数: {SIMULATION_STEPS}")
    print(f"====================\n")

    for p_idx, p in enumerate(p_list):
        cmdp = TwoSourceMPRCMDP(
            p1_only=p,
            p2_only=p,
            p1_both=p / 2,
            p2_both=p / 2,
            Fmax1=F1_max,
            Fmax2=F2_max,
        )

        # print(f"\n===== p = {p:.1f} =====")

        # ----- 1) LP 最优 -----
        try:
            X, pi_lp, mu_lp, C_lp, F1_lp, F2_lp = solve_cmdp_via_lp(cmdp)
            (c_mean, c_std), (f1_mean, f1_std), (f2_mean, f2_std) = simulate_policy_multiple(cmdp, pi_lp, mu0=mu_lp)
        except Exception as e:
            print(f"LP求解失败: {e}")
            C_lp = float('inf')
            c_mean = c_std = f1_mean = f1_std = f2_mean = f2_std = float('inf')

        lp_C_th.append(C_lp)
        lp_C_mc_mean.append(c_mean)
        lp_C_mc_std.append(c_std)
        lp_F1_mc_mean.append(f1_mean)
        lp_F1_mc_std.append(f1_std)
        lp_F2_mc_mean.append(f2_mean)
        lp_F2_mc_std.append(f2_std)

        # ----- 2) 双重二分 RVI -----
        try:
            bisection_res = cmdp.double_bisection_rvi()
            mu_bis, C_bisection_th, F1_bisection_th, F2_bisection_th = eval_policy_theoretical(
                cmdp, bisection_res['mix_policy']
            )
            (c_mean, c_std), (f1_mean, f1_std), (f2_mean, f2_std) = simulate_policy_multiple(
                cmdp, bisection_res['mix_policy'], mu0=mu_bis)
        except Exception as e:
            print(f"双重二分RVI求解失败: {e}")
            C_bisection_th = float('inf')
            c_mean = c_std = f1_mean = f1_std = f2_mean = f2_std = float('inf')

        bisection_C_th.append(C_bisection_th)
        bisection_C_mc_mean.append(c_mean)
        bisection_C_mc_std.append(c_std)
        bisection_F1_mc_mean.append(f1_mean)
        bisection_F1_mc_std.append(f1_std)
        bisection_F2_mc_mean.append(f2_mean)
        bisection_F2_mc_std.append(f2_std)

        # ----- 3) 固定策略 -----
        try:
            pi_fix = build_fixed_policy(cmdp)
            mu_fix, C_fix_th, F1_fix_th, F2_fix_th = eval_policy_theoretical(cmdp, pi_fix)
            (c_mean, c_std), (f1_mean, f1_std), (f2_mean, f2_std) = simulate_policy_multiple(cmdp, pi_fix, mu0=mu_fix)
        except Exception as e:
            print(f"固定策略求解失败: {e}")
            C_fix_th = float('inf')
            c_mean = c_std = f1_mean = f1_std = f2_mean = f2_std = float('inf')

        fix_C_th.append(C_fix_th)
        fix_C_mc_mean.append(c_mean)
        fix_C_mc_std.append(c_std)
        fix_F1_mc_mean.append(f1_mean)
        fix_F1_mc_std.append(f1_std)
        fix_F2_mc_mean.append(f2_mean)
        fix_F2_mc_std.append(f2_std)

        # ----- 4) DPP算法 -----
        # print(f"[DPP]     运行中...", end="")
        dpp_cae_list, dpp_f1_list, dpp_f2_list = [], [], []

        # 多次运行DPP算法取平均
        for run in range(SIMULATION_TIMES):
            dpp = LowComplexityDPP(cmdp, V1=100.0, V2=100.0)
            avg_cae, avg_freq = dpp.run_simulation(
                num_steps=SIMULATION_STEPS,
                burn_in=SIMULATION_BURN_IN
            )
            dpp_cae_list.append(avg_cae)
            dpp_f1_list.append(avg_freq[0])
            dpp_f2_list.append(avg_freq[1])

        # 计算均值和标准差
        dpp_C_mc_mean.append(np.mean(dpp_cae_list))
        dpp_C_mc_std.append(np.std(dpp_cae_list))
        dpp_F1_mc_mean.append(np.mean(dpp_f1_list))
        dpp_F1_mc_std.append(np.std(dpp_f1_list))
        dpp_F2_mc_mean.append(np.mean(dpp_f2_list))
        dpp_F2_mc_std.append(np.std(dpp_f2_list))

        # print(f" 完成! CAE={dpp_C_mc_mean[-1]:.4f}±{dpp_C_mc_std[-1]:.4f}")
        #
        # # 打印当前p值的所有结果
        # print(f"[LP]      C={lp_C_th[-1]:.4f} (MC={lp_C_mc_mean[-1]:.4f}±{lp_C_mc_std[-1]:.4f})")
        # print(f"[双二分RVI] C={bisection_C_th[-1]:.4f} (MC={bisection_C_mc_mean[-1]:.4f}±{bisection_C_mc_std[-1]:.4f})")
        # print(f"[Fixed]   C={fix_C_th[-1]:.4f} (MC={fix_C_mc_mean[-1]:.4f}±{fix_C_mc_std[-1]:.4f})")
        # print(f"[DPP]     C= N/A (MC={dpp_C_mc_mean[-1]:.4f}±{dpp_C_mc_std[-1]:.4f})")
        # 打印当前 p 的所有结果（新增F1/F2打印）
        print("\n--- Performance Summary for p = {:.1f} ---".format(p))

        print(f"[LP]       C={lp_C_th[-1]:.4f} "
              f"(MC={lp_C_mc_mean[-1]:.4f}±{lp_C_mc_std[-1]:.4f}), "
              f"F1={lp_F1_mc_mean[-1]:.3f}, F2={lp_F2_mc_mean[-1]:.3f}")

        print(f"[RVI]      C={bisection_C_th[-1]:.4f} "
              f"(MC={bisection_C_mc_mean[-1]:.4f}±{bisection_C_mc_std[-1]:.4f}), "
              f"F1={bisection_F1_mc_mean[-1]:.3f}, F2={bisection_F2_mc_mean[-1]:.3f}")

        print(f"[Fixed]    C={fix_C_th[-1]:.4f} "
              f"(MC={fix_C_mc_mean[-1]:.4f}±{fix_C_mc_std[-1]:.4f}), "
              f"F1={fix_F1_mc_mean[-1]:.3f}, F2={fix_F2_mc_mean[-1]:.3f}")

        print(f"[DPP]      C= N/A "
              f"(MC={dpp_C_mc_mean[-1]:.4f}±{dpp_C_mc_std[-1]:.4f}), "
              f"F1={dpp_F1_mc_mean[-1]:.3f}, F2={dpp_F2_mc_mean[-1]:.3f}")

    # ========================================================
    # 绘制包含DPP算法的对比图
    # ========================================================
    p_vals = p_list

    # 图1: 平均CAE对比（带误差棒）
    plt.figure(figsize=(12, 7))

    # 理论值
    plt.plot(p_vals, lp_C_th, 'b-o', linewidth=2, markersize=8, label='LP (Theory)')
    plt.plot(p_vals, bisection_C_th, 'r-s', linewidth=2, markersize=8, label='Dual Bisec RVI (Theory)')
    plt.plot(p_vals, fix_C_th, 'g-^', linewidth=2, markersize=8, label='FPS (Theory)')

    # 蒙特卡洛均值 + 误差棒
    plt.errorbar(p_vals, lp_C_mc_mean, yerr=lp_C_mc_std, fmt='b--o', linewidth=1.5,
                 markersize=6, capsize=4, label='LP (Monte Carlo)')
    plt.errorbar(p_vals, bisection_C_mc_mean, yerr=bisection_C_mc_std, fmt='r--s', linewidth=1.5,
                 markersize=6, capsize=4, label='Dual Bisec RVI (Monte Carlo)')
    plt.errorbar(p_vals, fix_C_mc_mean, yerr=fix_C_mc_std, fmt='g--^', linewidth=1.5,
                 markersize=6, capsize=4, label='FPS (Monte Carlo)')
    plt.errorbar(p_vals, dpp_C_mc_mean, yerr=dpp_C_mc_std, fmt='m--d', linewidth=2,
                 markersize=8, capsize=5, label='DPP (Online)')

    plt.xlabel("Channel Success Probability (p_only)", fontsize=12)
    plt.ylabel("Average CAE", fontsize=12)
    plt.title(f"Average CAE Comparison (F1_max={F1_max}, F2_max={F2_max})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(p_vals, [f'{x:.1f}' for x in p_vals])
    plt.tight_layout()
    plt.savefig('cae_comparison_with_dpp.png', dpi=300)
    plt.show()

    # 图2: 传感器1传输频率对比
    plt.figure(figsize=(12, 7))
    plt.errorbar(p_vals, lp_F1_mc_mean, yerr=lp_F1_mc_std, fmt='b-o', linewidth=2,
                 markersize=8, capsize=4, label='LP')
    plt.errorbar(p_vals, bisection_F1_mc_mean, yerr=bisection_F1_mc_std, fmt='r-s', linewidth=2,
                 markersize=8, capsize=4, label='Dual Bisec RVI')
    plt.errorbar(p_vals, fix_F1_mc_mean, yerr=fix_F1_mc_std, fmt='g-^', linewidth=2,
                 markersize=8, capsize=4, label='FPS')
    plt.errorbar(p_vals, dpp_F1_mc_mean, yerr=dpp_F1_mc_std, fmt='m-d', linewidth=2,
                 markersize=8, capsize=5, label='DPP')
    plt.axhline(F1_max, color='k', linestyle='--', alpha=0.7,
                label=f'Constraint (F1_max={F1_max})')
    plt.xlabel("Channel Success Probability (p_only)", fontsize=12)
    plt.ylabel("Transmission Frequency of Sensor 1", fontsize=12)
    plt.title("Transmission Frequency Comparison - Sensor 1", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(p_vals, [f'{x:.1f}' for x in p_vals])
    plt.tight_layout()
    plt.savefig('f1_comparison_with_dpp.png', dpi=300)
    plt.show()

    # 图3: 传感器2传输频率对比
    plt.figure(figsize=(12, 7))
    plt.errorbar(p_vals, lp_F2_mc_mean, yerr=lp_F2_mc_std, fmt='b-o', linewidth=2,
                 markersize=8, capsize=4, label='LP')
    plt.errorbar(p_vals, bisection_F2_mc_mean, yerr=bisection_F2_mc_std, fmt='r-s', linewidth=2,
                 markersize=8, capsize=4, label='Dual Bisec RVI')
    plt.errorbar(p_vals, fix_F2_mc_mean, yerr=fix_F2_mc_std, fmt='g-^', linewidth=2,
                 markersize=8, capsize=4, label='FPS')
    plt.errorbar(p_vals, dpp_F2_mc_mean, yerr=dpp_F2_mc_std, fmt='m-d', linewidth=2,
                 markersize=8, capsize=5, label='DPP')
    plt.axhline(F2_max, color='k', linestyle='--', alpha=0.7,
                label=f'Constraint (F2_max={F2_max})')
    plt.xlabel("Channel Success Probability (p_only)", fontsize=12)
    plt.ylabel("Transmission Frequency of Sensor 2", fontsize=12)
    plt.title("Transmission Frequency Comparison - Sensor 2", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(p_vals, [f'{x:.1f}' for x in p_vals])
    plt.tight_layout()
    plt.savefig('f2_comparison_with_dpp.png', dpi=300)
    plt.show()

    # 图4: 性能差距分析
    plt.figure(figsize=(12, 7))

    # 计算相对性能差距（相对于LP最优）
    dpp_gap_percent = []
    bisection_gap_percent = []
    fix_gap_percent = []

    for i in range(len(p_vals)):
        if lp_C_th[i] > 0 and lp_C_th[i] != float('inf'):
            dpp_gap = (dpp_C_mc_mean[i] - lp_C_th[i]) / lp_C_th[i] * 100
            bisection_gap = (bisection_C_th[i] - lp_C_th[i]) / lp_C_th[i] * 100
            fix_gap = (fix_C_th[i] - lp_C_th[i]) / lp_C_th[i] * 100

            dpp_gap_percent.append(dpp_gap)
            bisection_gap_percent.append(bisection_gap)
            fix_gap_percent.append(fix_gap)
        else:
            dpp_gap_percent.append(0)
            bisection_gap_percent.append(0)
            fix_gap_percent.append(0)

    plt.plot(p_vals, dpp_gap_percent, 'm-d', linewidth=3, markersize=10, label='DPP vs LP Optimal')
    plt.plot(p_vals, bisection_gap_percent, 'r-s', linewidth=2, markersize=8, label='Dual Bisec RVI vs LP Optimal')
    plt.plot(p_vals, fix_gap_percent, 'g-^', linewidth=2, markersize=8, label='FPS vs LP Optimal')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("Channel Success Probability (p_only)", fontsize=12)
    plt.ylabel("Performance Gap (%)", fontsize=12)
    plt.title("Performance Gap Relative to LP Optimal Solution", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(p_vals, [f'{x:.1f}' for x in p_vals])
    plt.tight_layout()
    plt.savefig('performance_gap_analysis.png', dpi=300)
    plt.show()

    # ========================================================
    # 性能汇总表
    # ========================================================
    print("\n" + "=" * 130)
    print("性能汇总表")
    print("=" * 130)
    print(f"{'p_only':<8} {'LP CAE':<10} {'RVI CAE':<12} {'Fixed CAE':<12} {'DPP CAE':<12} "
          f"{'RVI-LP(%)':<10} {'DPP-LP(%)':<10} {'Fixed-LP(%)':<10}")
    print("-" * 130)

    for i, p in enumerate(p_vals):
        if lp_C_th[i] == float('inf'):
            continue

        # 计算相对性能差距
        rvi_gap = (bisection_C_th[i] - lp_C_th[i]) / lp_C_th[i] * 100 if lp_C_th[i] != 0 else 0
        dpp_gap = (dpp_C_mc_mean[i] - lp_C_th[i]) / lp_C_th[i] * 100 if lp_C_th[i] != 0 else 0
        fix_gap = (fix_C_th[i] - lp_C_th[i]) / lp_C_th[i] * 100 if lp_C_th[i] != 0 else 0

        print(f"{p:<8.1f} {lp_C_th[i]:<10.4f} {bisection_C_th[i]:<12.4f} {fix_C_th[i]:<12.4f} "
              f"{dpp_C_mc_mean[i]:<12.4f} {rvi_gap:<10.2f} {dpp_gap:<10.2f} {fix_gap:<10.2f}")

    print("=" * 130)
    print("说明:")
    print("1. LP: 线性规划最优解（理论下界）")
    print("2. RVI: 双重二分相对值迭代")
    print("3. DPP: 低复杂度在线漂移加惩罚算法")
    print("4. Fixed: 固定概率策略")
    print("=" * 130)


def analyze_dpp_parameters():
    """分析DPP算法参数敏感性"""
    print("\n" + "=" * 60)
    print("DPP算法参数敏感性分析")
    print("=" * 60)

    # 选择一个典型的信道条件 (p=0.5)
    cmdp = TwoSourceMPRCMDP(
        p1_only=0.5,
        p2_only=0.5,
        p1_both=0.25,
        p2_both=0.25,
        Fmax1=0.4,
        Fmax2=0.4
    )

    # 分析V参数影响
    V_values = [1, 5, 10, 50, 100, 200, 500, 1000]
    analyzer = DPPSweepAnalyzer()
    results = analyzer.analyze_v_parameter(cmdp, V_values, num_steps=2000, num_runs=3)

    # 绘制V参数敏感性分析
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. V对CAE的影响
    axes[0, 0].semilogx(results['V'], results['avg_cae'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('V Parameter', fontsize=12)
    axes[0, 0].set_ylabel('Average CAE', fontsize=12)
    axes[0, 0].set_title('Effect of V on CAE', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. V对传输频率的影响
    axes[0, 1].semilogx(results['V'], results['avg_freq1'], 'ro-', label='Sensor 1', linewidth=2, markersize=8)
    axes[0, 1].semilogx(results['V'], results['avg_freq2'], 'go-', label='Sensor 2', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.4, color='k', linestyle='--', label='Constraint')
    axes[0, 1].set_xlabel('V Parameter', fontsize=12)
    axes[0, 1].set_ylabel('Transmission Frequency', fontsize=12)
    axes[0, 1].set_title('Effect of V on Transmission Frequency', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. V对队列稳定性的影响
    axes[1, 0].semilogx(results['V'], results['queue_variance'], 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('V Parameter', fontsize=12)
    axes[1, 0].set_ylabel('Queue Variance', fontsize=12)
    axes[1, 0].set_title('Effect of V on Queue Stability', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. CAE-传输频率权衡
    axes[1, 1].plot(results['avg_freq1'], results['avg_cae'], 's-', label='Sensor 1', linewidth=2)
    axes[1, 1].plot(results['avg_freq2'], results['avg_cae'], '^-', label='Sensor 2', linewidth=2)
    axes[1, 1].set_xlabel('Transmission Frequency', fontsize=12)
    axes[1, 1].set_ylabel('Average CAE', fontsize=12)
    axes[1, 1].set_title('CAE-Transmission Frequency Trade-off', fontsize=14)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dpp_parameter_sensitivity.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 运行包含DPP的主分析
    main_with_dpp()

    # 运行DPP参数敏感性分析
    # analyze_dpp_parameters()