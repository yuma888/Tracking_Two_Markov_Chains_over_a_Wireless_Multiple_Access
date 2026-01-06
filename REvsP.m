% ============================================================
%   MATLAB Figure Generator for Source Scheduling Dynamics
%   数据来源：flat_source_dynamics.mat
%   Q1 = slow source, Q2 = fast source
%   绘制共 5 张图，每张遵循 IEEE 双栏格式 (无标题版本)
% ============================================================
clear; clc; close all;

mat_file = 'result/REvsP.mat';
data = load(mat_file);

p = data.p;

algos = {'LP','RVI','FPS','DPP'};
metrics = {'avg_re','freq1','freq2','single1','single2','both','none',...
           's1_src1','s1_src2','s2_src1','s2_src2'};

for i = 1:length(algos)
    algo = algos{i};
    for j = 1:length(metrics)
        key = sprintf('%s_%s', algo, metrics{j});
        eval(sprintf('%s_%s = data.%s;', algo, metrics{j}, key));
    end
end

% =========================== IEEE 图配置 ===========================
figW = 6.5;
figH = 5.0;
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
set(0,'DefaultFigureColor','white');

% ========================= 空心绘图函数 ==============================
function plot_style_hollow(x, y, color, marker, linestyle)
    if nargin < 5, linestyle = '-'; end
    plot(x, y, ...
        'Color', color, ...
        'LineStyle', linestyle, ...
        'LineWidth', 1.3, ...
        'Marker', marker, ...
        'MarkerSize', 6, ...
        'MarkerEdgeColor', color, ...
        'MarkerFaceColor', 'w');
end

% ========================= 导出函数 ==============================
function export_ieee_pdf(figHandle, fileName, figW, figH, paperW, paperH)
    set(figHandle,'PaperUnits','inches');
    set(figHandle,'PaperSize',[paperW paperH]);
    set(figHandle,'PaperPosition',[(paperW-figW)/2,(paperH-figH)/2,figW,figH]);
    print(figHandle, fileName, '-dpdf','-painters','-r300');
end

% ============================================================
% FIGURE 1 — Average re vs p
% ============================================================
fig1 = figure; hold on; grid on; box on;

plot_style_hollow(p, LP_avg_re,  'b','o');
plot_style_hollow(p, RVI_avg_re, 'r','s');
plot_style_hollow(p, FPS_avg_re, 'g','^');
plot_style_hollow(p, DPP_avg_re, 'm','d','-');

xlabel('$p$','Interpreter','latex');
ylabel('$E(\pi)$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');

legend({'LP','RVI','FPS','DPP'},'Location','best');

export_ieee_pdf(fig1,'figures/RE_p',figW,figH,paperW,paperH);


% ============================================================
% FIGURE 2 — F1 / F2
% ============================================================
fig2 = figure;

subplot(2,1,1); hold on; grid on; box on;
plot_style_hollow(p, LP_freq1,  'b','o');
plot_style_hollow(p, RVI_freq1, 'r','s');
plot_style_hollow(p, FPS_freq1, 'g','^');
plot_style_hollow(p, DPP_freq1, 'm','d','-');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$F_1$','Interpreter','latex');
xlabel('$p$','Interpreter','latex');
xlim([min(p) max(p)]);
ylim([0 0.55]);

subplot(2,1,2); hold on; grid on; box on;
plot_style_hollow(p, LP_freq2,  'b','o');
plot_style_hollow(p, RVI_freq2, 'r','s');
plot_style_hollow(p, FPS_freq2, 'g','^');
plot_style_hollow(p, DPP_freq2, 'm','d','-');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$F_2$','Interpreter','latex');
xlabel('$p$','Interpreter','latex');
xlim([min(p) max(p)]);
ylim([0 0.55]);

export_ieee_pdf(fig2,'figures/RE_F1_F2_vs_p',figW,figH,paperW,paperH);


% ============================================================
% FIGURE 3 — Single / Dual
% ============================================================
fig3 = figure;

subplot(2,1,1); hold on; grid on; box on;
plot_style_hollow(p, LP_single1+LP_single2,'b','o');
plot_style_hollow(p, RVI_single1+RVI_single2,'r','s');
plot_style_hollow(p, FPS_single1+FPS_single2,'g','^');
plot_style_hollow(p, DPP_single1+DPP_single2,'m','d','-');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$F_{\mathrm{single}}$','Interpreter','latex','FontSize',11,'FontName','Times New Roman'); ylim([0 1.1]);
xlabel('$p$','Interpreter','latex');

subplot(2,1,2); hold on; grid on; box on;
plot_style_hollow(p, LP_both,'b','o');
plot_style_hollow(p, RVI_both,'r','s');
plot_style_hollow(p, FPS_both,'g','^');
plot_style_hollow(p, DPP_both,'m','d','--');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$F_{\mathrm{both}}$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');
xlabel('$p$','Interpreter','latex');

export_ieee_pdf(fig3,'figures/RE_single_dual_vs_p',figW,figH,paperW,paperH);


% ============================================================
% FIGURE 4 — S1 scheduling Q1/Q2
% ============================================================
fig4 = figure;

subplot(2,1,1); hold on; grid on; box on;
plot_style_hollow(p, LP_s1_src1,'b','o');
plot_style_hollow(p, RVI_s1_src1,'r','s');
plot_style_hollow(p, FPS_s1_src1,'g','^');
plot_style_hollow(p, DPP_s1_src1,'m','d','-');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$a_1^1$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');
xlabel('$p$','Interpreter','latex');

subplot(2,1,2); hold on; grid on; box on;
plot_style_hollow(p, LP_s1_src2,'b','o');
plot_style_hollow(p, RVI_s1_src2,'r','s');
plot_style_hollow(p, FPS_s1_src2,'g','^');
plot_style_hollow(p, DPP_s1_src2,'m','d','-');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$a_2^1$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');
xlabel('$p$','Interpreter','latex');

export_ieee_pdf(fig4,'figures/RE_S1_p',figW,figH,paperW,paperH);


% ============================================================
% FIGURE 5 — S2 scheduling Q1/Q2
% ============================================================
fig5 = figure;

subplot(2,1,1); hold on; grid on; box on;
plot_style_hollow(p, LP_s2_src1,'b','o');
plot_style_hollow(p, RVI_s2_src1,'r','s');
plot_style_hollow(p, FPS_s2_src1,'g','^');
plot_style_hollow(p, DPP_s2_src1,'m','d','-');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$a_1^2$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');
xlabel('$p$','Interpreter','latex');

subplot(2,1,2); hold on; grid on; box on;
plot_style_hollow(p, LP_s2_src2,'b','o');
plot_style_hollow(p, RVI_s2_src2,'r','s');
plot_style_hollow(p, FPS_s2_src2,'g','^');
plot_style_hollow(p, DPP_s2_src2,'m','d','-');
legend({'LP','RVI','FPS','DPP'}, 'Location','best', 'Interpreter','latex');
ylabel('$a_2^2$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');
xlabel('$p$','Interpreter','latex');

export_ieee_pdf(fig5,'figures/RE_S2_p',figW,figH,paperW,paperH);

