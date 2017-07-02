% | j = 1 |

% plot(is09_200_al_X, is09_200_al_Y); hold on;
% plot(is09_200_rdal_q1_X, is09_200_rdal_q1_Y); hold on;
% plot(is09_200_odal_fix_q1_X, is09_200_odal_fix_q1_Y); hold on;
% plot(is09_200_odal_upd_q1_X, is09_200_odal_upd_q1_Y);
% 
% xlabel('# Human Annotations');
% ylabel('UAR [%]');
% % title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Conf=Platt');
% hleg = legend('SAL', 'rDAL', 'oDAL (fixed reliability)', 'oDAL (updated reliability)', 'Location','southeastoutside');
% 
% hlt = text(...
%     'Parent', hleg.DecorationContainer, ...
%     'String', 'Type of Active Learning', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'bottom', ...
%     'Position', [0.5, 1.05, 0], ...
%     'Units', 'normalized');
% 
% set(gca,'gridlinestyle','--');
% grid on;
% set(gcf,'units','points','position',[400,400,680,180]);


% | j = 2 |

% plot(is09_200_al_X, is09_200_al_Y); hold on;
% plot(is09_200_rdal_q2_X, is09_200_rdal_q2_Y); hold on;
% plot(is09_200_odal_fix_q2_X, is09_200_odal_fix_q2_Y); hold on;
% plot(is09_200_odal_upd_q2_X, is09_200_odal_upd_q2_Y);
% 
% xlabel('# Human Annotations');
% ylabel('UAR [%]');
% % title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Conf=Platt');
% hleg = legend('SAL', 'rDAL', 'oDAL (fixed reliability)', 'oDAL (updated reliability)', 'Location','southeastoutside');
% 
% hlt = text(...
%     'Parent', hleg.DecorationContainer, ...
%     'String', 'Type of Active Learning', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'bottom', ...
%     'Position', [0.5, 1.05, 0], ...
%     'Units', 'normalized');
% 
% set(gca,'gridlinestyle','--');
% grid on;
% set(gcf,'units','points','position',[400,400,680,180]);


% | j = 3 |

plot(is09_200_al_X, is09_200_al_Y); hold on;
plot(is09_200_rdal_q1_X, is09_200_rdal_q1_Y); hold on;
plot(is09_200_odal_fix_q1_X, is09_200_odal_fix_q1_Y); hold on;
plot(is09_200_odal_upd_q1_X, is09_200_odal_upd_q1_Y);

xlabel('# Human Annotations');
ylabel('UAR [%]');
% title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Conf=Platt');
hleg = legend('SAL', 'rDAL', 'oDAL (fixed reliability)', 'oDAL (updated reliability)', 'Location','southeastoutside');

hlt = text(...
    'Parent', hleg.DecorationContainer, ...
    'String', 'Type of Active Learning', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'Position', [0.5, 1.05, 0], ...
    'Units', 'normalized');

set(gca,'gridlinestyle','--');
grid on;
set(gcf,'units','points','position',[400,400,680,180]);


% plot(is09_200_odal_fix_q2_X, is09_200_odal_fix_q2_Y); hold on;
% plot(is09_500_odal_fix_q2_X, is09_500_odal_fix_q2_Y);
% 
% xlabel('# Human Annotations');
% ylabel('UAR [%]');
% % title('Self-training; SVM (sklearn) C=0.05; #inst=200; Norm=Z-std; Conf=Platt');
% hleg = legend('200', '500', 'Location','southeastoutside');
% 
% hlt = text(...
%     'Parent', hleg.DecorationContainer, ...
%     'String', '# instances', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'bottom', ...
%     'Position', [0.5, 1.05, 0], ...
%     'Units', 'normalized');
% 
% set(gca,'gridlinestyle','--');
% grid on;
% set(gcf,'units','points','position',[400,400,680,180]);