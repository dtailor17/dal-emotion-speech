% load('data.mat', 'x_data', 'y_avg')
load('IS09.mat', 'x_data', 'y_avg')
plot(x_data, y_avg)
xlabel('# Machine Labelled Instances')
ylabel('UAR [%]')
set(gca,'gridlinestyle','--')
grid on
