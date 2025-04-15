function [X, loss, GT_error, timer] = ScaledGD(X_GT_full, X, X_GT_Omega, Idx_Omega, opts)

    if ~isfield( opts, 'maxiter');  opts.maxiter = 100;     end
    if ~isfield( opts, 'cg');       opts.cg = false;         end
    if ~isfield( opts, 'tol');      opts.tol = 1e-6;        end
    
    alpha = opts.alpha;
    rho = opts.rho;
    tol = opts.tol;
    
    n = X.size;
    order = X.order;
    
    disp('ScaledGD Algorithm Starts!')
    loss = zeros(opts.maxiter, 1);
    GT_error = zeros(opts.maxiter, 1);
    timer = zeros(opts.maxiter, 1);
    X_full = ttm(X.core, X.U, 1:order);
    norm_X_GT = norm(X_GT_full);

    U_plus = cell(order, 1);
    U_pinv = cell(order, 1);
    for i=1:opts.maxiter
        tic;
        Grad = tensor(zeros(n));
        Grad(Idx_Omega) = X_full(Idx_Omega) - X_GT_Omega;
        for k = 1:order
            Zm = double(tenmat(Grad, k));
            Um = double(tenmat(ttm(X.core, X.U, -k), k, 't'));
            U_plus{k} = X.U{k} - alpha/rho*Zm*Um/(Um'*Um);
            U_pinv{k} = X.U{k}/(X.U{k}'*X.U{k});
        end
        X.core = X.core - alpha/rho * ttm(Grad, U_pinv, 't');
        X.U = U_plus;
        timer(i) = toc;

        X_full = ttm(X.core, X.U, 1:order);
        loss(i) = 0.5 * norm(X_full(Idx_Omega) - X_GT_Omega)^2;
        GT_error(i) = norm(X_full - X_GT_full)/norm_X_GT;
        fprintf('Iteration: %d, Error: %.4e\n', i, GT_error(i));
        if GT_error(i) < tol
            break
        end

    end
