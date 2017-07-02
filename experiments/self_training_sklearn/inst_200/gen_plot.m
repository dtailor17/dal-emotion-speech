load('data_200.mat');

% Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Class unbalanced (DIST VS PROB)
% plot(x_data, cwNone_dist_std);
% hold on;
% plot(x_data, cwNone_prob_std);
% xlabel('# Machine Labelled Instances');
% ylabel('UAR [%]');
% title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Class unbalanced')
% hleg = legend('Distance-to-hyperplane', 'Platt Calibration', 'Location','southeast');
% 
% hlt = text(...
%     'Parent', hleg.DecorationContainer, ...
%     'String', 'Confidence Measure', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'bottom', ...
%     'Position', [0.5, 1.05, 0], ...
%     'Units', 'normalized');
% 
% set(gca,'gridlinestyle','--');
% grid on;
% set(gcf,'units','points','position',[400,400,550,180])

% Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Min-Max; Class unbalanced
% plot(x_data, cwNone_prob_minmax);
% xlabel('# Machine Labelled Instances');
% ylabel('UAR [%]');
% title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Min-Max; Class unbalanced')
% hleg = legend('Platt Calibration', 'Location','southeast');
% 
% hlt = text(...
%     'Parent', hleg.DecorationContainer, ...
%     'String', 'Confidence Measure', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'bottom', ...
%     'Position', [0.5, 1.05, 0], ...
%     'Units', 'normalized');
% 
% set(gca,'gridlinestyle','--');
% grid on;
% set(gcf,'units','points','position',[400,400,565,180])

% Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Classes weighted (DIST VS PROB)
% plot(x_data, cwBal_dist_std);
% hold on;
% plot(x_data, cwBal_prob_std);
% xlabel('# Machine Labelled Instances');
% ylabel('UAR [%]');
% title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Classes weighted');
% hleg = legend('Distance-to-hyperplane', 'Platt Calibration', 'Location','southeastoutside');
% 
% hlt = text(...
%     'Parent', hleg.DecorationContainer, ...
%     'String', 'Confidence Measure', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'bottom', ...
%     'Position', [0.5, 1.05, 0], ...
%     'Units', 'normalized');
% 
% set(gca,'gridlinestyle','--');
% grid on;
% set(gcf,'units','points','position',[400,400,680,180]);

% Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Classes weighted (DIST VS PROB)
plot(x_data, cwNone_prob_std); hold on;
plot(x_data, cwBal_prob_std); hold on;
plot(x_data, cwOver_prob_std);
xlabel('# Machine Labelled Instances');
ylabel('UAR [%]');
title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Conf=Platt');
hleg = legend('None', 'Class weighted (SVM)', 'Oversampled', 'Location','southeastoutside');

hlt = text(...
    'Parent', hleg.DecorationContainer, ...
    'String', 'Resampling Method', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'Position', [0.5, 1.05, 0], ...
    'Units', 'normalized');

set(gca,'gridlinestyle','--');
grid on;
set(gcf,'units','points','position',[400,400,680,180]);
