% ============================================================
%    IEEE 双栏图配置 (读取Python按算法拆分的.mat数据 + 绘图)
%    数据来源：Python仿真保存的 RE_vs_fmax.mat 文件
%    字体：普通文本=Times New Roman，公式保留LaTeX格式
% ============================================================
clear; clc; close all;

% --------------------------
% 1. 读取Python的.mat数据
% --------------------------
mat_file = 'result/re_vs_fmax.mat';
data = load(mat_file);

Fmax = data.Fmax;          % X轴：0.1~1.0（10个点）

% RE数据
lp_RE = data.lp_re;      % LP算法RE
rvi_RE = data.rvi_re;    % RVI算法RE
fps_RE = data.fps_re;    % FPS算法RE
dpp_RE = data.dpp_re;    % DPP算法RE

% F1/F2频率数据
lp_F1 = data.lp_f1;        % LP算法F1频率
rvi_F1 = data.rvi_f1;      % RVI算法F1频率
fps_F1 = data.fps_f1;      % FPS算法F1频率
dpp_F1 = data.dpp_f1;      % DPP算法F1频率

lp_F2 = data.lp_f2;        % LP算法F2频率
rvi_F2 = data.rvi_f2;      % RVI算法F2频率
fps_F2 = data.fps_f2;      % FPS算法F2频率
dpp_F2 = data.dpp_f2;      % DPP算法F2频率

% 传输模式数据
lp_single1 = data.lp_single1;  % LP单发1
rvi_single1 = data.rvi_single1;% RVI单发1
fps_single1 = data.fps_single1;% FPS单发1
dpp_single1 = data.dpp_single1;% DPP单发1

lp_single2 = data.lp_single2;  % LP单发2
rvi_single2 = data.rvi_single2;% RVI单发2
fps_single2 = data.fps_single2;% FPS单发2
dpp_single2 = data.dpp_single2;% DPP单发2

lp_both = data.lp_both;        % LP双发
rvi_both = data.rvi_both;      % RVI双发
fps_both = data.fps_both;      % FPS双发
dpp_both = data.dpp_both;      % DPP双发

% 计算单发求和（single1 + single2）
lp_single_sum = lp_single1 + lp_single2;
rvi_single_sum = rvi_single1 + rvi_single2;
fps_single_sum = fps_single1 + fps_single2;
dpp_single_sum = dpp_single1 + dpp_single2;

% --------------------------
% 2. IEEE标准配置（兼容所有MATLAB版本）
% --------------------------
figW   = 6.5;     % IEEE double-column width (inches)
figH   = 5.0;     % 适配分栏子图
margin = 0.05;     
paperW = figW + 2*margin;
paperH = figH + 2*margin;

% 全局样式 (IEEE 推荐 + Times New Roman字体)
set(0,'DefaultAxesFontSize',11);
set(0,'DefaultTextFontSize',11);
set(0,'DefaultLineLineWidth',1.2);
set(0,'DefaultAxesLineWidth',0.9);
set(0,'DefaultFigureUnits','inches');
set(0,'DefaultAxesFontName','Times New Roman');  % 坐标轴字体：新罗马
set(0,'DefaultTextFontName','Times New Roman');  % 文本字体：新罗马
set(0,'DefaultAxesTickLabelInterpreter','latex'); % 刻度标签LaTeX解释器
set(0,'DefaultLegendInterpreter','latex');       % 图例LaTeX解释器

% --------------------------
% 3. 图1: RE性能对比
% --------------------------
fig1 = figure('Color','white'); hold on; grid on; box on;
% 核心：设置坐标轴字体为Times New Roman（替代无效的XTickLabelFontName）
set(gca,'GridAlpha',0.3,'GridLineStyle','-','FontName','Times New Roman');

% 绘制RE曲线
plot(Fmax, lp_RE,     'b-o',  'MarkerSize',7, 'LineWidth',1.2, 'DisplayName','LP');
plot(Fmax, rvi_RE,    'r-s',  'MarkerSize',7, 'LineWidth',1.2, 'DisplayName','RVI');
plot(Fmax, fps_RE,    'g-^',  'MarkerSize',7, 'LineWidth',1.2, 'DisplayName','FPS');
plot(Fmax, dpp_RE,    'm-d', 'MarkerSize',7, 'LineWidth',1.5, 'DisplayName','DPP');

% 标签：普通文本=Times New Roman，公式保留LaTeX
xlabel(['$F_{\max}$'],...
    'Interpreter','latex','FontSize',11,'FontName','Times New Roman')
ylabel('$E(\pi)$',...
    'Interpreter','latex','FontSize',11,'FontWeight','bold','FontName','Times New Roman');
lgd1 = legend('Location','best','Interpreter','latex');
set(lgd1,'Box','on','LineWidth',0.9,'FontSize',10,'FontName','Times New Roman');
xlim([0.1 1.0]);

% 自适应Y轴范围（排除NaN）
all_RE = [lp_RE, rvi_RE, fps_RE, dpp_RE];
all_RE_valid = all_RE(~isnan(all_RE));
ylim([0.35, 0.9]);

% 导出
export_ieee_pdf(fig1,'figures/RE_Fmax',figW,figH,paperW,paperH);

% --------------------------
% 4. 图2: 传感器F1/F2发射频率对比
% --------------------------
fig2 = figure('Color','white');

% 上子图: F1 (Sensor 1) 发射频率
subplot(2,1,1); hold on; grid on; box on;
set(gca,'GridAlpha',0.3,'GridLineStyle','-','FontName','Times New Roman');

plot(Fmax, lp_F1,    'b-o',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','LP');
plot(Fmax, rvi_F1,   'r-s', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','RVI');
plot(Fmax, fps_F1,   'g-^',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','FPS');
plot(Fmax, dpp_F1,   'm-d', 'MarkerSize',6, 'LineWidth',1.5, 'DisplayName','DPP');
plot(Fmax, Fmax,     'k--',  'LineWidth',1.0, 'DisplayName','$F_{\max}$');

% 标签：分离文本和公式
xlabel(['$F_{\max}$'],...
    'Interpreter','latex','FontSize',11,'FontName','Times New Roman')
ylabel('$F^1$','Interpreter','latex','FontSize',11,'FontWeight','bold','FontName','Times New Roman');
lgd2 = legend('Location','best','Interpreter','latex');
set(lgd2,'Box','on','LineWidth',0.9,'FontSize',9,'FontName','Times New Roman');
xlim([0.1 1.0]);
ylim([0 1.1]);

% 下子图: F2 (Sensor 2) 发射频率
subplot(2,1,2); hold on; grid on; box on;
set(gca,'GridAlpha',0.3,'GridLineStyle','-','FontName','Times New Roman');

plot(Fmax, lp_F2,    'b-o',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','LP');
plot(Fmax, rvi_F2,   'r-s', 'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','RVI');
plot(Fmax, fps_F2,   'g-^',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','FPS');
plot(Fmax, dpp_F2,   'm-d', 'MarkerSize',6, 'LineWidth',1.5, 'DisplayName','DPP');
plot(Fmax, Fmax,     'k--',  'LineWidth',1.0, 'DisplayName','$F_{\max}$');

% 标签：分离文本和公式
xlabel(['$F_{\max}$'],...
    'Interpreter','latex','FontSize',11,'FontName','Times New Roman')
ylabel('$F^2$','Interpreter','latex','FontSize',11,'FontWeight','bold','FontName','Times New Roman');
lgd3 = legend('Location','best','Interpreter','latex');
set(lgd3,'Box','on','LineWidth',0.9,'FontSize',9,'FontName','Times New Roman');
xlim([0.1 1.0]);
ylim([0 1.1]);

% 统一样式
set(findall(fig2,'Type','Axes'),'FontSize',10,'FontName','Times New Roman');
% 导出
export_ieee_pdf(fig2,'figures/RE_F1F2Fmax',figW,figH,paperW,paperH);

% --------------------------
% 5. 图3: 纯单发+纯双发对比
% --------------------------
fig3 = figure('Color','white');

% 上子图: 纯单发模式 (single1 + single2 求和)
subplot(2,1,1); hold on; grid on; box on;
set(gca,'GridAlpha',0.3,'GridLineStyle','-','FontName','Times New Roman');

plot(Fmax, lp_single_sum,  'b-o',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','LP');
plot(Fmax, rvi_single_sum, 'r-s',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','RVI');
plot(Fmax, fps_single_sum, 'g-^',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','FPS');
plot(Fmax, dpp_single_sum, 'm-d', 'MarkerSize',6, 'LineWidth',1.5, 'DisplayName','DPP');

% 标签：分离文本和公式
xlabel(['$F_{\max}$'],...
    'Interpreter','latex','FontSize',11,'FontName','Times New Roman')
ylabel('$F_{\mathrm{single}}$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');
lgd4 = legend('Location','best','Interpreter','latex');
set(lgd4,'Box','on','LineWidth',0.9,'FontSize',9,'FontName','Times New Roman');
xlim([0.1 1.0]);
ylim([0 1.1]);  % 适配单发求和范围

% 下子图: 纯双发模式 (both列数据)
subplot(2,1,2); hold on; grid on; box on;
set(gca,'GridAlpha',0.3,'GridLineStyle','-','FontName','Times New Roman');

plot(Fmax, lp_both,  'b-o',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','LP');
plot(Fmax, rvi_both, 'r-s',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','RVI');
plot(Fmax, fps_both, 'g-^',  'MarkerSize',6, 'LineWidth',1.2, 'DisplayName','FPS');
plot(Fmax, dpp_both, 'm-d', 'MarkerSize',6, 'LineWidth',1.5, 'DisplayName','DPP');

% 标签：分离文本和公式
xlabel(['$F_{\max}$'],...
    'Interpreter','latex','FontSize',11,'FontName','Times New Roman')
ylabel('$F_{\mathrm{both}}$','Interpreter','latex','FontSize',11,'FontName','Times New Roman');
lgd5 = legend('Location','best','Interpreter','latex');
set(lgd5,'Box','on','LineWidth',0.9,'FontSize',9,'FontName','Times New Roman');
xlim([0.1 1.0]);
ylim([0 1.1]);

% 统一样式
set(findall(fig3,'Type','Axes'),'FontSize',10,'FontName','Times New Roman');
% 导出
export_ieee_pdf(fig3,'figures/RE_SingleBothFmax',figW,figH,paperW,paperH);

% --------------------------
% 6. 输出完成提示 + 数据验证
% --------------------------
fprintf('✅ 从按算法拆分的.mat文件读取数据并生成三张IEEE标准图完成：\n');
fprintf('- Fig1_RE_Performance.pdf (RE性能对比)\n');
fprintf('- Fig2_Sensor_F1F2_Frequency.pdf (F1/F2频率)\n');
fprintf('- Fig3_Single1Single2_Sum_Plus_Dual.pdf (单发+双发)\n');

% ============================================================
%  子函数：IEEE 标准 PDF/EPS 导出
% ============================================================
function export_ieee_pdf(figHandle, fileName, figW, figH, paperW, paperH)
    set(figHandle,'PaperUnits','inches');
    set(figHandle,'PaperSize',[paperW paperH]);
    set(figHandle,'PaperPosition',[(paperW-figW)/2, (paperH-figH)/2, figW, figH]);
    set(figHandle,'Units','inches','Position',[2, 2, figW, figH]);
    
    % 最终兜底：所有非公式文本强制为Times New Roman
    text_objs = findall(figHandle, 'Type', 'Text');
    for i = 1:length(text_objs)
        set(text_objs(i), 'FontName', 'Times New Roman');
    end
    
    % 高质量矢量导出 (兼容LaTeX/Word)
    print(figHandle, fileName, '-dpdf', '-painters', '-r300');
end