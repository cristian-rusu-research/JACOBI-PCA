function [positions, values, approx_error, tus, Ubar, S, diagonal] = algoritm1_block_random(S, b, p)
%% Algorithm 1, but selection of indices is uniformly random
%% Input:
% S - a symmetric matrix of size dxd
% b - the number of generalized block transformations to use for the approximation of the eigenspace
% p - size of the eigenspace

%% Output:
% The g generalized Givens transformations:
% positions - the two indices (i,j) where the transformation operates
% values - the four values of the transformations
% approx_error - the approximation error, as defined in the paper
% tus - the total running time
% Ubar - the approximate eigenspace
% S - the updated S, i.e., S^(g)
% diagonal - trace performance measure

tic;
[d, ~] = size(S);

%% basic sanity check
if (d <= 1) || (b < 1)
    positions = []; values = []; tus = toc;
    return;
end
if norm(S-S', 'fro') >= 10e-7
    error('S has to be symmetric');
end

%% make sure we have a positive integer
b = round(b);

%% errors
diagonal = [];
approx_error = [];

%% vector that will store the indices (i,j) and the values of the transformations for each of the g Givens transformations
positions = zeros(2*p, b);
values = zeros(4*p^2, b);

%% initialization of each Givens transformation
for kk = b:-1:1
    indices = [1:p randsample(p+1:d, p)];
    indices = unique(sort([1:p indices]));
    
    if (length(indices) < 2*p)
        other_indices = setdiff(1:d, indices');
        indices = [indices other_indices(1:(2*p-length(indices)))];
        indices = unique(sort(indices));
    end

    %% compute the optimum orthogonal transformation on the optimum indices
    [Vv, Dd] = eig(S(indices, indices));
    [~, inds] = sort(diag(Dd), 'descend');
    Vv = Vv(:, inds);
    GG = Vv';

    %% save the block Givens transformation
    positions(:, kk) = indices;
    values(:, kk) = vec(GG);
    
    GG = speye(d);
    GG(indices, indices) = reshape(values(:, kk), 2*p, 2*p);

    %% update the working matrix
    S = GG*S*GG';
    
    diagonal = [diagonal trace(S(1:p,1:p))];
    approx_error = [approx_error norm_off_diagonal(S)];
end

%% explictly build Ubar
Ubar = eye(d);
for k = 1:b
    Ubar(positions(:, k), positions(:, k)) = reshape(values(:, k), 2*p, 2*p)'*Ubar(positions(:, k), positions(:, k));
end

%% time everything
tus = toc;
