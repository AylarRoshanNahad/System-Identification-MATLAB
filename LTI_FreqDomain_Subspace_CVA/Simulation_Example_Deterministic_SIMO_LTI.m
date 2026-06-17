% =========================================================================
% SIMO Frequency-Domain State-Space System Identification using CVA
% =========================================================================

clear; clc; close all;

%% 1. Define the True System (Discrete-Time)

fprintf('Initializing True System...\n');

n = 2; % Number of states
p = 2; % Number of outputs
m = 1; % Number of inputs

A = [0, 1; -0.2, -0.5];
B = [0; 1];
C = [1, 0; 0, 1];
D = zeros(p, m);
x0 = [0; 0];

Ts = 1; % DT sampling time
sys_true = ss(A, B, C, D, Ts);

%% 2. Data Collection (Simulating the Frequency Response)

fprintf('Collecting Frequency Response Data...\n');

Ti = 5:2:150;           % Excitation periods
wi = 2*pi ./ Ti;        % Excitation frequencies (rad/s)
num_tests = length(Ti);

window_start_mul = 5;   % Periods to wait for steady state
window_length_mul = 1;  % Periods to evaluate for FFT

Y_resp_2D = zeros(p, num_tests);

% Simulate and extract steady-state frequency response
for i = 1:num_tests
    [~, ~, Y_resp_2D(:, i)] = LTI_freq(Ti(i), A, B, C, D, x0, window_start_mul, window_length_mul);
end

% CRITICAL FORMATTING: MATLAB's idfrd requires a 3D array for frequency data
% Dimensions must strictly be: [Outputs (p), Inputs (m), Frequencies]
Y_resp_3D = reshape(Y_resp_2D, [p, m, num_tests]);

%% 3. Subspace System Identification (N4SID - CVA)
% Reference Note: 
% n4sid takes the frequency data
% computes the Markov parameters (impulse response)
% builds a Hankel matrix
% uses SVD to extract the A, B, C, D matrices based on the CVA weighting scheme

fprintf('\n================ RUNNING CVA ESTIMATION =============== \n');


% Create the frequency response data object
data_fd = idfrd(Y_resp_3D, wi, Ts);

% Configure the subspace algorithm options
opt = n4sidOptions('Focus', 'simulation', 'EnforceStability', true, 'Display', 'off');
opt.N4Weight = 'CVA'; 

n_est = 2; % Prescribed state dimension
sys_hat = n4sid(data_fd, n_est, opt, 'Feedthrough', true);

% Hankel Singular Values to check state dim
hsv = hsvd(sys_hat);
fprintf('Hankel Singular Values \n');
disp(hsv);
fprintf('\n');
% Determine n based on Singular values

%% 4. Model Evaluation & Plotting
fprintf('\n================ FIT EVALUATION ======================= \n');

% Generate the frequency response of the newly identified model
resp_id_3D = freqresp(sys_hat, wi);

% Metric 1: Overall Variance Accounted For (VAF)
% How much of the actual dynamics (variance) did the model capture?
diff_mag = abs(Y_resp_3D(:)) - abs(resp_id_3D(:));
vaf = max(0, (1 - var(diff_mag) / var(abs(Y_resp_3D(:)))) * 100);
fprintf('Overall Frequency-Domain VAF: %6.2f%%\n', vaf);

% Metric 2: Visual Bode Plot Comparison
% Extract 1D magnitude arrays for plotting
mag_data1 = squeeze(abs(Y_resp_3D(1, 1, :)));
mag_data2 = squeeze(abs(Y_resp_3D(2, 1, :)));
mag_id1   = squeeze(abs(resp_id_3D(1, 1, :)));
mag_id2   = squeeze(abs(resp_id_3D(2, 1, :)));

figure('Name', 'CVA Frequency Response Verification', 'Position', [100, 100, 900, 400]);

% Output 1 Plot
subplot(1, 2, 1);
plot(wi, mag_data1, 'k-', 'LineWidth', 2); hold on;
plot(wi, mag_id1, 'b--', 'LineWidth', 2);
title('Output 1 (y_1) Frequency Response');
xlabel('Frequency \omega (rad/s)'); 
ylabel('Magnitude |G(j\omega)|');
legend('True System', 'Identified (CVA)', 'Location', 'best');
grid on;

% Output 2 Plot
subplot(1, 2, 2);
plot(wi, mag_data2, 'k-', 'LineWidth', 2); hold on;
plot(wi, mag_id2, 'r--', 'LineWidth', 2);
title('Output 2 (y_2) Frequency Response');
xlabel('Frequency \omega (rad/s)'); 
ylabel('Magnitude |G(j\omega)|');
legend('True System', 'Identified (CVA)', 'Location', 'best');
grid on;

%% 5. Numerical Diagnostics
fprintf('\n================ MODEL DIAGNOSTICS ==================== \n');

% 1. Hankel Singular Values 
hsv = hsvd(sys_hat);
fprintf('Hankel Singular Values:\n');
disp(hsv(1:length(hsv))'); 

% Plot HSV
figure('Name', 'Hankel Singular Values', 'Position', [1050, 100, 500, 400]);
bar(1:length(hsv), hsv, 'FaceColor', [0.2 0.6 0.8]);
title('Hankel Singular Values (Log Scale)');
xlabel('State Index'); ylabel('Energy'); grid on;

% 2. Condition Number of the A matrix
% Reference Note: High condition number (>1e4) means the state space basis
% is skewed and will accumulate floating-point errors during long simulations.
cond_A = cond(sys_hat.A);
fprintf('Condition Number of A Matrix: %.2f\n', cond_A);
if cond_A > 1e4
    fprintf('  -> WARNING: A matrix is poorly conditioned. Consider balreal().\n');
else
    fprintf('  -> SUCCESS: A matrix is well-conditioned.\n');
end

% 3. Pole Comparison (True vs Identified)
% the eigenvalues (poles) must match if the dynamics are identical.
true_poles = eig(A);
est_poles = pole(sys_hat);

fprintf('\nPole Comparison (True vs Estimated):\n');
for idx = 1:n
    fprintf('  Pole %d: True = %6.4f %s %6.4fj | Est = %6.4f %s %6.4fj\n', ...
        idx, real(true_poles(idx)), char(sign(imag(true_poles(idx)))+44), abs(imag(true_poles(idx))), ...
        real(est_poles(idx)), char(sign(imag(est_poles(idx)))+44), abs(imag(est_poles(idx))));
end

max_pole_mag = max(abs(est_poles));
fprintf('\nMaximum Discrete Pole Magnitude: %.4f\n', max_pole_mag);
if max_pole_mag >= 1
    fprintf('  -> WARNING: Identified system is UNSTABLE.\n');
else
    fprintf('  -> SUCCESS: Identified system is STABLE.\n');
end
fprintf('======================================================= \n');

%% 6. Time-Domain Verification (Step Response)

t_sim = 0:Ts:30; % Simulate for 30 seconds

% Calculate step responses
[y_true_step, t_out] = step(sys_true, t_sim);
[y_id_step, ~]       = step(sys_hat, t_sim);

figure('Name', 'Time-Domain Verification: Step Response', 'Position', [100, 50, 900, 400]);

% Output 1 Step Plot
subplot(1, 2, 1);
stairs(t_out, y_true_step(:, 1), 'k-', 'LineWidth', 2); hold on;
stairs(t_out, y_id_step(:, 1), 'b--', 'LineWidth', 2);
title('Output 1 (y_1) Step Response');
xlabel('Time (s)'); ylabel('Amplitude');
legend('True System', 'Identified (CVA)', 'Location', 'best'); grid on;

% Output 2 Step Plot
subplot(1, 2, 2);
stairs(t_out, y_true_step(:, 2), 'k-', 'LineWidth', 2); hold on;
stairs(t_out, y_id_step(:, 2), 'r--', 'LineWidth', 2);
title('Output 2 (y_2) Step Response');
xlabel('Time (s)'); ylabel('Amplitude');
legend('True System', 'Identified (CVA)', 'Location', 'best'); grid on;

fprintf('======================================================= \n');

%% HELPER FUNCTIONS
function [U_resp, Y_resp, X_resp] = LTI_freq(T, A, B, C, D, x0, window_start_mul, window_length_mul)

    % 1. Dimensions (SIMO assumption)
    n = size(A, 1);
    p = size(C, 1);
    
    % 2. Time boundaries
    T_mul = window_length_mul * T;
    start_idx = window_start_mul * T;
    t_max = start_idx + T_mul + 1;
    t = 1:t_max;
    
    % 3. Excitation signal 
    % u(t) = 2*cos(omega*t) ensures the fundamental FFT peak has amplitude 1
    u = 2 * cos((2 * pi / T) * t);
    
    % 4. Pre-allocation
    x = zeros(n, t_max + 1);
    x(:, 1) = x0;  % Fixed: Initialize the entire first column vector
    y = zeros(p, t_max);
    
    % 5. Time-domain simulation
    for i = 1:t_max
        x(:, i+1) = A * x(:, i) + B * u(i);
        y(:, i)   = C * x(:, i) + D * u(i);
    end
    
    % 6. Steady-state extraction (Fixed: Added row dimension ':' for matrices)
    u_ss = u(start_idx : start_idx + T_mul - 1);
    x_ss = x(:, start_idx : start_idx + T_mul - 1);
    y_ss = y(:, start_idx : start_idx + T_mul - 1);
    
    num_samples = numel(u_ss);
    
    % 7. FFT calculation
    % Fixed: Added '[], 2' to compute FFT along the rows (time dimension)
    U_FS = fft(u_ss, [], 2) / num_samples;
    X_FS = fft(x_ss, [], 2) / num_samples;
    Y_FS = fft(y_ss, [], 2) / num_samples;
    
    % 8. Frequency Indexing
    % Without fftshift, DC is at index 1.
    % The fundamental frequency is exactly 'window_length_mul' bins away.
    freq_idx = window_length_mul + 1;
    
    
    % 9. Output assignment
    U_resp = U_FS(:, freq_idx);
    X_resp = X_FS(:, freq_idx);
    Y_resp = Y_FS(:, freq_idx);
    
end
