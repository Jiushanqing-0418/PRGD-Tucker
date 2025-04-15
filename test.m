% Comparison of Algorithms for Low Multilinear Rank Tensor Completion
addpath('./tensor_toolbox-v3.1')

% Initialization
clear; clc;
randn('state',27);
rand('state',27);

% Parameters
d = 3;          % tensor order
n = 100;        % dimension
r = 5;          % rank
OS = 10;        % oversampling ratio
nn = n * ones(1, d);
rr = r * ones(1, d);

% Sampling parameters
dims = r^d + d*(n*r - r*r);
sizeOmega = min(OS*dims, 0.99*prod(nn));
rho = sizeOmega / prod(nn);
fprintf('Oversample ratio: %d, Sampling ratio: %.2f\n', OS, rho);

% Generate ground truth tensor
Idx_Omega = randsample(prod(nn), sizeOmega, false);
X_GT = TUCKER_rand(rr, nn, d);
X_GT_full = ttm(X_GT.core, X_GT.U, 1:d);
X_GT_Omega = X_GT_full(Idx_Omega);

% Initialization
X0 = Spectral_initial_offdiag(X_GT_Omega, Idx_Omega, rr, nn, d, rho);
opts_tucker = struct('maxiter', 10000, 'tol', 1e-8, 'rho', rho, 'alpha', 0);

%% Algorithm Comparisons
% RGD
opts_tucker.alpha = 3.70;
[X_RGD, ~, GT_error_RGD, timer_RGD] = RGD(X_GT_full, X0, X_GT_Omega, Idx_Omega, opts_tucker);

% PRGD
epsilon = 5e-4;
opts_tucker.alpha = 1.18;
[X_PRGD, ~, GT_error_PRGD, timer_PRGD] = PRGD(X_GT_full, X0, X_GT_Omega, Idx_Omega, opts_tucker, epsilon);

% ScaledGD
opts_tucker.alpha = 0.135;
[X_ScaledGD, ~, GT_error_ScaledGD, timer_ScaledGD] = ScaledGD(X_GT_full, X0, X_GT_Omega, Idx_Omega, opts_tucker);

fprintf('Runtimes (s):\nRGD: %.2f\nPRGD: %.2f\nScaledGD: %.2f\n', ...
        sum(timer_RGD), sum(timer_PRGD), sum(timer_ScaledGD));

%% Visualization
close all
colors = [0.85 0.325 0.098;  % Orange
          0.601 0.401 0.631;   % Purple
          0 0.447 0.741];  % Blue

% Error vs Iterations
figure(1);
semilogy(GT_error_PRGD, '-', 'color', colors(1,:), 'linewidth', 2); hold on;
semilogy(GT_error_ScaledGD, '-', 'color', colors(2,:), 'linewidth', 2);
semilogy(GT_error_RGD, '-', 'color', colors(3,:), 'linewidth', 2);
grid on;
xlabel('Iterations'); ylabel('Relative Error');
legend({'PRGD', 'ScaledGD', 'RGD'});
title(sprintf('OS = %d, d = %d, n = %d, r = %d', OS, d, n, r));

% Error vs CPU Time
figure(2);
semilogy(cumsum(timer_PRGD), GT_error_PRGD, '-', 'color', colors(1,:), 'linewidth', 2); hold on;
semilogy(cumsum(timer_ScaledGD), GT_error_ScaledGD, '-', 'color', colors(2,:), 'linewidth', 2);
semilogy(cumsum(timer_RGD), GT_error_RGD, '-', 'color', colors(3,:), 'linewidth', 2);
grid on;
xlabel('CPU time'); ylabel('Relative Error');
legend({'PRGD', 'ScaledGD', 'RGD'});
title(sprintf('OS = %d, d = %d, n = %d, r = %d', OS, d, n, r));

%% Functions 
function X = TUCKER_rand(rr, nn, d)
    X.core = tensor(rand(rr));
    X.U = cell(1, d);
    for i = 1:d
        [X.U{i}, ~] = qr(rand(nn(i), rr(i)), 0);
    end
    X.size = nn;
    X.rank = rr;
    X.order = d;
end

function X = Spectral_initial_offdiag(X_GT_Omega, Idx_Omega, rr, nn, d, rho)
    Y = tensor(zeros(nn));
    Y(Idx_Omega) = X_GT_Omega;
    
    % Spectral initialization
    U0 = cell(d, 1);
    for k = 1:d
        Ym = double(tenmat(Y, k));
        G = Ym*Ym';
        [U0{k}, ~] = eigs(G - diag(diag(G)), rr(k));
    end
    
    X.core = ttm(Y/rho, U0, 't');
    X.U = U0;
    X.size = nn;
    X.rank = rr;
    X.order = d;
end