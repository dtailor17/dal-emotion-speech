load('matlab.mat');

plot(x_rdal, y_rdal_avg); hold on;
plot(x_odal_fix, y_odal_fix_avg); hold on;
plot(x_odal_upd, y_odal_upd_avg);
xlabel('# Labelled Instances');
ylabel('UAR [%]');
hleg = legend('rDAL', 'oDAl (fix)', 'oDAL (updated)', 'Location', 'southeast');

hlt = text(...
    'Parent', hleg.DecorationContainer, ...
    'String', 'Learning Type', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'Position', [0.5, 1.05, 0], ...
    'Units', 'normalized');

set(gca,'gridlinestyle','--');
grid on;
set(gcf,'units','points','position',[400,400,680,180]);
