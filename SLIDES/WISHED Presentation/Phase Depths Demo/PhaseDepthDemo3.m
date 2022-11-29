clear;

lambda = 854e-9;
d = 0:10e-9:10e-3;

phi = (2*pi/lambda) * d;
phi_wrapped = angle(exp(j*phi));

y = cos(phi);

lims = [0 5];

figure(1);
clf;
subplot(2,1,1);
plot(d*1e6, phi_wrapped);
xlim(lims);
grid on;
title('Optical Path Length versus Phase Shift', 'FontSize', 30);
xlabel('Optical Path Length ({\mu}m)', 'FontSize', 24);
ylabel('Phase Shift (radians)', 'FontSize', 24);
% subplot(2,1,2);
% plot(d*1e6, y);
% xlim(lims);
% grid on;