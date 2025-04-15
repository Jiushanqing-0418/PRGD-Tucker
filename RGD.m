function [X, loss, GT_error, timer] = RGD(X_GT_full, X, X_GT_Omega, Idx_Omega, opts)
% This code is to implement the RGD algorithm for low multilinear rank tensor completion
% X_GT_full: The ground truth low multilinear rank full tensor
% X: Initial point
% X_GT_Omega: The observation of X
% Idx_Omega: The index of sampling set Omega
% opts: The hyper-parameters of RGD algorithm

if ~isfield( opts, 'maxiter');  opts.maxiter = 100;     end
if ~isfield( opts, 'cg');       opts.cg = false;         end
if ~isfield( opts, 'tol');      opts.tol = 1e-6;        end

alpha = opts.alpha;
tol = opts.tol;

disp('RGD Algorithm Starts!')
order = X.order;
loss = zeros(opts.maxiter, 1);
GT_error = zeros(opts.maxiter, 1);
timer = zeros(opts.maxiter, 1);
X_full = ttm(X.core, X.U, 1:order);
norm_X_GT = norm(X_GT_full);
for i = 1:opts.maxiter
    tic;
    grad = X_full(Idx_Omega) - X_GT_Omega;
    Proj_cmpnts = Tangent_proj(grad, X, Idx_Omega);
    X = tangentAdd(X, Proj_cmpnts, alpha);
    timer(i) = toc;

    X_full = ttm(X.core, X.U, 1:order);
    loss(i) = 0.5 * norm(X_full(Idx_Omega) - X_GT_Omega)^2;
    GT_error(i) = norm(X_full - X_GT_full)/norm_X_GT;
    fprintf('Iteration: %d, Error: %.4e\n', i, GT_error(i));
    if GT_error(i) < tol
        break
    end

end


end
function Proj_cmpnts = Tangent_proj(grad, X, Idx_Omega)
order = X.order;
n = X.size;
Grad = tensor(zeros(n));
Grad(Idx_Omega) = grad;

W = cell(1, order);
D = ttm(Grad, X.U, 1:order, 't');

for i = 1:order
    K = ttm(Grad, X.U, -i, 't');
    L = ttm(K, (eye(n(i)) - X.U{i}*X.U{i}'), i);
    W{i} = double(tenmat(L, i))/double(tenmat(X.core, i));
end
Proj_cmpnts.D = D;
Proj_cmpnts.W = W;
end

function X = tangentAdd(X, Proj_cmpnts, alpha)
order = X.order;
n = X.size;
r = X.rank;

core = X.core;
D = Proj_cmpnts.D;
W = Proj_cmpnts.W;

core_tilde = tensor(zeros(2 * r));
% Block-1: core - alpha*D
sBlock = double(core - alpha * D);
for i = 1:order
    sBlock = cat(i, sBlock, zeros(size(sBlock)));
end
core_tilde = core_tilde + tensor(sBlock);

% Adjacent Blocks - alpha*core
for i = 1:order
    sBlock = double( - alpha * core);
    modes = 1:order;
    modes(i) = [];
    for j=modes
        sBlock = cat(j, sBlock, zeros(size(sBlock)));
    end
    sBlock = cat(i, zeros(size(sBlock)), sBlock);
    core_tilde = core_tilde + tensor(sBlock);
end
for i = 1:order
    [Q, R] = qr([X.U{i}, W{i}], 0);
    X.U{i} = Q;
    core_tilde = ttm(core_tilde, R, i);
end
core_t_HOSVD = hosvd(core_tilde, r);
X.core = core_t_HOSVD.core;
for i=1:order
    X.U{i} = X.U{i} * core_t_HOSVD.U{i};
end
end

function X = hosvd(X_full, rr)
d = length(rr);
nn = size(X_full);
U = cell(1, d);
for i = 1:d
    X_mat = double(tenmat(X_full, i));
    [U{i}, ~, ~] = svds(X_mat, rr(i));
end
core = ttm(X_full, U, 1:d, 't');
X.U = U;
X.core = core;
X.size = nn;
X.rank = rr;
X.order = d;
end
