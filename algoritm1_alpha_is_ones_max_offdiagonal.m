function [positions, values, approx_error, tus, Ubar, S, diagonal] = algoritm1_alpha_is_ones_max_offdiagonal(S, g, p)
%% Jacobi selection (largest off-diagonal element)
%% Input:
% S - a symmetric matrix of size dxd
% g - the number of generalized transformations to use for the approximation of the eigenspace
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
if (d <= 1) || (g < 1)
    positions = []; values = []; tus = toc;
    return;
end
if norm(S-S', 'fro') >= 10e-7
    error('S has to be symmetric');
end

%% make sure we have a positive integer
g = round(g);

%% errors
diagonal = [];
approx_error = [];

%% vector that will store the indices (i,j) and the values of the transformations for each of the g Givens transformations
positions = zeros(2, g);
values = zeros(4, g);

%% compute all scores
scores = zeros(p, d);
for i = 1:p
    for j = p+1:d
        scores(i, j) = abs(S(i,j));
    end
end

%% initialization of each Givens transformation
for kk = g:-1:1
    %% check where the maximum scores is, to find the optimum indices
    [~, index_nuc] = max(scores(:));
    [i_nuc, j_nuc] = ind2sub([p d], index_nuc);

    %% compute the optimum orthogonal transformation on the optimum indices
    [Vv, Dd] = eig(S([i_nuc j_nuc], [i_nuc j_nuc]));
    [~, inds] = sort(diag(Dd), 'descend');
    Vv = Vv(:, inds);
    GG = Vv';

    %% save the Givens transformation
    positions(1, kk) = i_nuc;
    positions(2, kk) = j_nuc;
    values(:, kk) = vec(GG);

    %% update the working matrix
    S = applyGTransformOnRightTransp(S, i_nuc, j_nuc, values(:, kk));
    S = applyGTransformOnLeft(S, i_nuc, j_nuc, values(:, kk));
    
    diagonal = [diagonal trace(S(1:p,1:p))];
    approx_error = [approx_error norm_off_diagonal(S)];

    %% update the scores only for the coordinates that were selected, everything else is the same
    for j = p+1:d
       scores(i_nuc, j) = abs(S(i_nuc, j));
    end
    for i = 1:p
        scores(i, j_nuc) = abs(S(i, j_nuc));
    end
end

%% explictly build Ubar
Ubar = eye(d);
for k = 1:g
    aux = values(2, k);
    values(2, k) = values(3, k);
    values(3, k) = aux;
    
    Ubar = applyGTransformOnLeft(Ubar, positions(1, k), positions(2, k), values(:, k));
end

%% time everything
tus = toc;
