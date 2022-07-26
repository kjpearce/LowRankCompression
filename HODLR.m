% HODLR Compression 

% Outputs: 

% mainIndCell: holds the full "binary" tree of indices starting at root
    % each cell row is the full partition of nodes for that level
    % i.e. mainIndCell{level,1} is a cell array of size num_boxes x 1s (number
    % of boxes in a given level: 2^(level-1))
    % for each tau (parent of box siblings)
        % alpha index cell rows correspond to cell indices that are 1 mod 2
        % beta index cell rows correspond to cell indices that are 0 mod 2

% MatStrc: structure that holds basis matrices for each level (excluding root)
    % at each level (row of struct), every field is a cell (num_parents x 1) 
    % A or B cells have alpha or beta matrices, respectively: row #s paired
    % e.g. first off diagonal block A(I_alpha,I_beta) =
    % MatStrc(1).Ualphas{1}*MatStrc(1).Diag_ab{1}*[MatStrc(1).V_b{1}]'

% DiagBoxes: diagonal blocks corresponding to leaves

clear 
close all
rng default 

% sample points
n = 400;
theta = 0.1;
x = rand(n,1);
x = sort(x,'ascend');

% kernel matrix
A = zeros(n,n); 
for row = 1:n
    for col = 1:n
        A(row,col) = exp(-(abs(x(row)-x(col)))/theta); % kernel function
    end
end

L = 3;   % number of levels  (not including root 0)
k = 2;   % target rank of off-diagonals
p = 5;   % oversampling parameter
r = k+p; % rank plus oversampling parameter (# cols in random matrices)

mainIndCell = cell(L+1,1);  % holds each level's partitions starting at level 0 (root)
mainIndCell{1,1} = {[1:n]}; % root indices
A_l = zeros(n,n);           % compressed off-diagonals

MatStrc = struct; % hold all basis matrices by level, excluding root

for level = 2:L+1
    num_child= 2^(level-1); % total number of children in level
    TempCell = cell(num_child,1); % stores indices of all children
    num_parents = num_child/2; % total number of parents in level
    counter = 1;
    
    % initialize Gamma1 and Gamma2
    Gamma1 = zeros(n,r);
    Gamma2 = zeros(n,r);

    PrevCell = mainIndCell{level-1,1}; % parents tau

    for parent = 1:num_parents
        temp = PrevCell{parent,1};          % grab parent indices from previous level
        [Ialpha, Ibeta] = IndexSplit(temp); % split parent indices into kids
        n_alph = length(Ialpha);
        n_beta = length(Ibeta);

        Gamma1(Ialpha,:) = randn(n_alph,r);
        Gamma2(Ibeta,:)  = randn(n_beta,r);

        TempCell{counter,1} = Ialpha; % alphas are 1 mod 2
        counter = counter + 1;

        TempCell{counter,1} = Ibeta;  % betas are 0 mod 2
        counter = counter + 1;
    end

    % TempCell now holds all children alpha, beta of all parents tau in level
    % Gamma1, Gamma2 filled in

    Y1 = A*Gamma2 - A_l*Gamma2; % samples for incoming basis matrices
    Y2 = A*Gamma1 - A_l*Gamma1;

    mainIndCell{level,1} = TempCell;
    
    child = 1;
    for parent = 1:num_parents % could re-write to go by children with mod 2 for parallelization?

        I_alpha = TempCell{child,1}; % child alpha
        child = child + 1;       

        I_beta = TempCell{child,1};  % child beta
        child = child + 1;

        if (length(I_alpha) < r) || (length(I_beta) < r)
            sprintf('Decrease r or L: length of index vectors too small at level %d',level)
            break
        end

        [U_alph,~] = qr(Y1(I_alpha,:),0);
        [U_beta,~] = qr(Y2(I_beta,:),0);

        Gamma1(I_alpha,:) = U_alph; % overwrite random matrices
        Gamma2(I_beta,:)  = U_beta;

    end

    Z1 = A'*Gamma2 - (A_l)'*Gamma2; % samples for outgoing basis matrices
    Z2 = A'*Gamma1 - (A_l)'*Gamma1;
    
    U_Acell = cell(num_parents,1); % holds alpha, beta U basis mats
    U_Bcell = cell(num_parents,1);

    D_ABcell = cell(num_parents,1); % holds alpha, beta B mats
    D_BAcell = cell(num_parents,1);

    V_Acell = cell(num_parents,1); % holds alpha, beta V mats
    V_Bcell = cell(num_parents,1);

    child = 1;
    for parent = 1:num_parents
        I_alpha = TempCell{child,1}; 
        child = child + 1;

        I_beta = TempCell{child,1};
        child = child + 1;

        [V_a, B_ba, Uhat_b] = svd(Z1(I_alpha,:),'econ');
        [V_b, B_ab, Uhat_a] = svd(Z2(I_beta,:),'econ');

        % grab old U's
        U_alpha = Gamma1(I_alpha,:);
        U_beta  = Gamma2(I_beta,:);
        
        % update U basis matrices
        U_a = U_alpha*Uhat_a; 
        U_b = U_beta*Uhat_b;

        % compress off-diagonals and store in A_l
        A_l(I_alpha,I_beta) = U_a*B_ab*V_b';
        A_l(I_beta,I_alpha) = U_b*B_ba*V_a';

        U_Acell{parent,1} = U_a;
        U_Bcell{parent,1} = U_b;

        V_Acell{parent,1} = V_a;
        V_Bcell{parent,1} = V_b;

        D_ABcell{parent,1} = B_ab;
        D_BAcell{parent,1} = B_ba;

    end
    
    MatStrc(level-1).Ualphas = U_Acell;
    MatStrc(level-1).Ubetas = U_Bcell;

    MatStrc(level-1).Valphas = V_Acell;
    MatStrc(level-1).Vbetas = V_Bcell;

    MatStrc(level-1).B_ab = D_ABcell;
    MatStrc(level-1).B_ba = D_BAcell;
end

% at this point, have off diagonals: now extract diagonals
leaf_inds  = mainIndCell{end}; % indices of leaves
num_leaves = length(leaf_inds); % number of leaves
n_max = 1;
for leaf = 1:num_leaves
    temp = leaf_inds{leaf};
    if length(temp) > n_max
        n_max = length(temp); % max length of index vec among leaves
    end
end

Gamma = zeros(n, n_max); % matrix for diagonals

for leaf = 1:num_leaves
    I_leaf = leaf_inds{leaf};
    n_leaf = length(I_leaf);
    Gamma(I_leaf,1:n_leaf) = eye(n_leaf); % fill in with identity
end

Y = A*Gamma - A_l*Gamma; % sampling matrix for diagonals

A_test = A_l;

DiagBoxes = cell(num_leaves,1); % store diagonal matrices
for leaf = 1:num_leaves
    I_leaf = leaf_inds{leaf};
    n_leaf = length(I_leaf);
    DiagBoxes{leaf,1} = Y(I_leaf,1:n_leaf);
    A_test(I_leaf,I_leaf) = Y(I_leaf,1:n_leaf);
end

approx_err = norm(A-A_test); 

clearvars -except x n r L k p theta mainIndCell MatStrc A_l A DiagBoxes A_test approx_err


