function [X, loss, GT_error, timer] = PRGD(X_GT_full, X, X_GT_Omega, Idx_Omega, opts, epsilon)
    % This code is to implement the PRGD algorithm for low multilinear rank tensor completion
    % X_GT_full: The ground truth low multilinear rank full tensor 
    % X: Initial point
    % X_GT_Omega: The observation of X
    % Idx_Omega: The index of sampling set Omega
    % opts: The hyper-parameters of PRGD algorithm
    % epsilon: The hyperparameter in precondition metric
    
    if ~isfield( opts, 'maxiter');  opts.maxiter = 100;     end
    if ~isfield( opts, 'cg');       opts.cg = false;         end
    if ~isfield( opts, 'tol');      opts.tol = 1e-6;        end
    
    alpha = opts.alpha;
    tol = opts.tol;
    
    disp('Preconditioned RGD Algorithm Starts!')
    order = X.order;
    loss = zeros(opts.maxiter, 1);
    GT_error = zeros(opts.maxiter, 1);
    timer = zeros(opts.maxiter, 1);
    X_full = ttm(X.core, X.U, 1:order);
    norm_X_GT = norm(X_GT_full);
    for i = 1:opts.maxiter
        tic;
        grad = X_full(Idx_Omega) - X_GT_Omega;
        G_pre = Pre_mtx2(grad, Idx_Omega, X.size, epsilon);
        X = orthogonalize_pre(X, G_pre);
        Proj_cmpnts_pre = Tangent_proj_pre(grad, G_pre, X, Idx_Omega); 
        X = tangentAdd(X, Proj_cmpnts_pre, alpha);
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
   
    
function G_pre = Pre_mtx2(grad, Idx_Omega, n, epsilon)
    % Compute the slice norm of the gradient
    order = length(n);
    G_pre = cell(1, order);
    Grad2 = tensor(zeros(n));
    Grad2(Idx_Omega) = grad.^2;
    Grad2 = double(Grad2);
    temp = Grad2;
    G_pre{1} = temp;
    for i=1:order-1
        temp = sum(temp,i);
        G_pre{i+1} = temp;
    end
    for i=order:-1:1
        for j=order:-1:i+1
            G_pre{i} = sum(G_pre{i}, j);
        end
        G_pre{i} = reshape(G_pre{i}, n(i), 1) + epsilon;
    end  
end

function X = orthogonalize_pre(X, G_pre)
    order = X.order;
    n = X.size;
    r = X.rank;
    for ii = 1:order
        G_ii = diag(G_pre{ii}.^(1/(2*order)));
        M_ii = sqrtm(X.U{ii}' * G_ii * X.U{ii});
        X.U{ii} = X.U{ii} * inv(M_ii);
        X.core = ttm(X.core, M_ii, ii);
    end
end

function Proj_cmpnts_pre = Tangent_proj_pre(grad, G_pre, X, Idx_Omega)
    order = X.order;
    n = X.size;
    Grad_pre = tensor(zeros(n));

    % pre of gradient
    G_pre_sqrt_inv = cell(1, order);
    for i = 1:order
        G_pre_sqrt_inv{i} = G_pre{i}.^(-1/(4*order));
    end
    G_pre_sqrt_inv_tensor = tensor(ktensor(G_pre_sqrt_inv));
    Grad_pre(Idx_Omega) = grad .* G_pre_sqrt_inv_tensor(Idx_Omega);

    % pre of factor matirces
    for i = 1:order
        G_i_sqrt = diag(G_pre{i}.^(1/(4*order)));
        X.U{i} = G_i_sqrt * X.U{i};
    end

    W = cell(1, order);
    D = ttm(Grad_pre, X.U, 1:order, 't');

    for i = 1:order
        K = ttm(Grad_pre, X.U, -i, 't');
        L = ttm(K, (eye(n(i)) - X.U{i}*X.U{i}'), i);
        W{i} = double(tenmat(L, i))/double(tenmat(X.core, i));
        G_i_sqrt_inv = sparse(1:n(i), 1:n(i), G_pre{i}.^(-1/(4*order)), n(i), n(i));
        W{i} = G_i_sqrt_inv * W{i};
    end
    Proj_cmpnts_pre.D = D;
    Proj_cmpnts_pre.W = W;
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

    % Adjacent Blocks -alpha*core,
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
    