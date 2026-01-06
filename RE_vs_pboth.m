% ============================================================
%    IEEE 双栏图配置 (读取Python按算法拆分的.mat数据 + 绘图)
%    数据来源：Python仿真保存的 re_vs_p_both.mat 文件
% ============================================================

clear; clc; close all;

% --------------------------
% 1. 读取Python的.mat数据
% --------------------------
mat_file = 'result/re_vs_p_both.mat';
data = load(mat_file);

p_both = data.p_both;    % X轴

% === re ===
lp_re  = data.lp_avg_re;
rvi_re = data.rvi_avg_re;
fps_re = data.fps_avg_re;
dpp_re = data.dpp_avg_re;

% === F1 / F2 ===
lp_F1  = data.lp_freq1;   rvi_F1 = data.rvi_freq1;
fps_F1 = data.fps_freq1;  dpp_F1 = data.dpp_freq1;

lp_F2  = data.lp_freq2;   rvi_F2 = data.rvi_freq2;
fps_F2 = data.fps_freq2;  dpp_F2 = data.dpp_freq2;

% === single1/2 & both ===
lp_s1 = data.lp_single1;     rvi_s1 = data.rvi_single1;
fps_s1 = data.fps_single1;   dpp_s1 = data.dpp_single1;

lp_s2 = data.lp_single2;     rvi_s2 = data.rvi_single2;
fps_s2 = data.fps_single2;   dpp_s2 = data.dpp_single2;

lp_both2 = data.lp_both;     rvi_both2 = data.rvi_both;
fps_both2 = data.fps_both;   dpp_both2 = data.dpp_both;

% 求和：纯单发 = single1 + single2
lp_single_sum  = lp_s1 + lp_s2;
rvi_single_sum = rvi_s1 + rvi_s2;
fps_single_sum = fps_s1 + fps_s2;
dpp_single_sum = dpp_s1 + dpp_s2;

% --------------------------
% 2. IEEE标准配置
% --------------------------
figW   = 6.5;
figH   = 5.0;
margin = 0.05;
paperW = figW + 2*margin;
paperH = figH + 2*margin;

set(0,'DefaultAxesFontSize',11);
set(0,'DefaultTextFontSize',11);
set(0,'DefaultLineLineWidth',1.2);
set(0,'DefaultAxesLineWidth',0.9);
set(0,'DefaultFigureUnits','inches');
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
set(0,'DefaultAxesTickLabelInterpreter','latex');
set(0,'DefaultLegendInterpreter','latex');


% ============================================================
% 3. 图1: re vs p_both
% ============================================================
fig1 = figure('Color','white'); hold on; grid on; box on;

plot(p_both, lp_re,  'b-o',  'MarkerSize',7, 'LineWidth',1.2, 'DisplayName','LP');
plot(p_both, rvi_re, 'r-s',  'MarkerSize',7, 'LineWidth',1.2, 'DisplayName','RVI');
plot(p_both, fps_re, 'g-^',  'MarkerSize',7, 'LineWidth',1.2, 'DisplayName','FPS');
plot(p_both, dpp_re, 'm-d', 'MarkerSize',7, 'LineWidth',1.5, 'DisplayName','DPP');

xlabel('$p_{\mathrm{both}}$', 'Interpreter','latex','FontSize',11);
ylabel('$C(\pi)$', ...
    'Interpreter','latex','FontSize',11,'FontWeight','bold');

lgd1 = legend('Location','best','Interpreter','latex');
set(lgd1,'Box','on','LineWidth',0.9,'FontSize',10);

xlim([min(p_both) max(p_both)]);
all_re = [lp_re; rvi_re; fps_re; dpp_re];
valid = all_re(~isnan(all_re));
ylim([0.2, 0.7]);

export_ieee_pdf(fig1,'figures/RE_vs_pboth',figW,figH,paperW,paperH);


% ============================================================
% 4. 图2: F1 / F2 发射频率
% ============================================================
fig2 = figure('Color','white');

% ---- 上子图: F1 ----
subplot(2,1,1); hold on; grid on; box on;

plot(p_both, lp_F1, 'b-o', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','LP');
plot(p_both, rvi_F1,'r-s', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','RVI');
plot(p_both, fps_F1,'g-^',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','FPS');
plot(p_both, dpp_F1,'m-d', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','DPP');

xlabel('$p_{\mathrm{both}}$', 'Interpreter','latex','FontSize',11);
ylabel('$F_1$', 'Interpreter','latex','FontSize',11,'FontWeight','bold');
legend('Location','best'); xlim([min(p_both) max(p_both)]); ylim([0.35 0.65]);

% ---- 下子图: F2 ----
subplot(2,1,2); hold on; grid on; box on;

plot(p_both, lp_F2, 'b-o', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','LP');
plot(p_both, rvi_F2,'r-s', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','RVI');
plot(p_both, fps_F2,'g-^',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','FPS');
plot(p_both, dpp_F2,'m-d', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','DPP');

xlabel('$p_{\mathrm{both}}$', 'Interpreter','latex','FontSize',11);
ylabel('$F_2$', 'Interpreter','latex','FontSize',11,'FontWeight','bold');
legend('Location','best'); xlim([min(p_both) max(p_both)]); ylim([0.45 0.65]);

export_ieee_pdf(fig2,'figures/RE_F1F2_vs_pboth',figW,figH,paperW,paperH);


% ============================================================
% 5. 图3: (single1+single2) 与 both
% ============================================================
fig3 = figure('Color','white');

% ---- 上子图：单发求和 ----
subplot(2,1,1); hold on; grid on; box on;

plot(p_both, lp_single_sum,  'b-o','LineWidth',1.2,'DisplayName','LP');
plot(p_both, rvi_single_sum, 'r-s','LineWidth',1.2,'DisplayName','RVI');
plot(p_both, fps_single_sum, 'g-^','LineWidth',1.2,'DisplayName','FPS');
plot(p_both, dpp_single_sum, 'm-d','LineWidth',1.5,'DisplayName','DPP');

xlabel('$p_{\mathrm{both}}$', 'Interpreter','latex','FontSize',11);
ylabel('$F_{\mathrm{single}}$', 'Interpreter','latex','FontSize',11);
legend('Location','best'); ylim([0.4 1.1]); xlim([min(p_both) max(p_both)]);

% ---- 下子图：双发 both ----
subplot(2,1,2); hold on; grid on; box on;

plot(p_both, lp_both2, 'b-o','LineWidth',1.2,'DisplayName','LP');
plot(p_both, rvi_both2,'r-s','LineWidth',1.2,'DisplayName','RVI');
plot(p_both, fps_both2,'g-^','LineWidth',1.2,'DisplayName','FPS');
plot(p_both, dpp_both2,'m-d','LineWidth',1.5,'DisplayName','DPP');

xlabel('$p_{\mathrm{both}}$', 'Interpreter','latex','FontSize',11);
ylabel('$F_{\mathrm{both}}$', 'Interpreter','latex','FontSize',11);
legend('Location','best'); ylim([0 0.45]); xlim([min(p_both) max(p_both)]);

export_ieee_pdf(fig3,'figures/RE_singleSum_and_both_vs_pboth',figW,figH,paperW,paperH);


fprintf("\n 成功从 re_vs_p_both.mat 读取数据，并绘制三张 IEEE 图！\n");


% ============================================================
%  子函数：IEEE 标准 PDF/EPS 导出
% ============================================================
function export_ieee_pdf(figHandle, fileName, figW, figH, paperW, paperH)
    set(figHandle,'PaperUnits','inches');
    set(figHandle,'PaperSize',[paperW paperH]);
    set(figHandle,'PaperPosition',[(paperW-figW)/2, (paperH-figH)/2, figW, figH]);
    set(figHandle,'Units','inches','Position',[2, 2, figW, figH]);

    text_objs = findall(figHandle, 'Type', 'Text');
    for i = 1:length(text_objs)
        set(text_objs(i), 'FontName', 'Times New Roman');
    end

    print(figHandle, fileName, '-dpdf', '-painters', '-r300');
end
