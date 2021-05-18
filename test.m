close all
clear
clc
format short

% size of symmetric matrix
n = 128;

% random symmetric matrix
A = randn(n);
A = A*A';

% perform the EVD
tic; [U, L] = eig(A); t_eig = toc;
[~, indices] = sort(diag(L));
L = L(indices, indices);
U = U(:, indices);

% number of principal components to recover
p = 20;

% number of basic transformations Gij
g = round(p*log2(n));

% block size, for the block algorithms
b = round(g/(p));

%% call algorithms
alpha = log2(p+1:-1:2);

[positions_alpha_is_one, values_alpha_is_one, approx_error_alpha_is_one, tus_alpha_is_one, Ubar_alpha_is_one, S_alpha_is_one, diagonal_alpha_is_one] = ...
    algoritm1_alpha_is_ones(A, g, p);

[positions_alpha_is_one_max_offdiagonal, values_alpha_is_one_max_offdiagonal, approx_error_alpha_is_one_max_offdiagonal, tus_alpha_is_one_max_offdiagonal, Ubar_alpha_is_one_max_offdiagonal, S_alpha_is_one_max_offdiagonal, diagonal_alpha_is_one_max_offdiagonal] = ...
    algoritm1_alpha_is_ones_max_offdiagonal(A, g, p);

[positions_alpha_is_updated, values_alpha_is_updated, approx_error_alpha_is_updated, tus_alpha_is_updated, Ubar_alpha_is_updated, S_alpha_is_updated, diagonal_alpha_is_updated] = ...
    algoritm1_alpha_is_updated(A, g, p);

[positions_block, values_block, approx_error_block, tus_block, Ubar_block, S_block, diagonal_block] = algoritm1_block(A, b, p);

[positions_block_max, values_block_max, approx_error_block_max, tus_block_max, Ubar_block_max, S_block_max, diagonal_block_max] = algoritm1_block_max(A, b, p);

[positions_block_random, values_block_random, approx_error_block_random, tus_block_random, Ubar_block_random, S_block_random, diagonal_block_random] = algoritm1_block_random(A, b, p);

[positions, values, approx_error, tus, Ubar, S, diagonal] = algoritm1(A, alpha, g, p);

%% plot results
total = trace(L(end-p+1:end, end-p+1:end));
figure;
plot(diagonal/total*100, 'k');
hold on; plot(diagonal_alpha_is_one/total*100, 'g');
%hold on; plot(diagonal_alpha_is_updated/total*100, 'b');
hold on; plot(diagonal_alpha_is_one_max_offdiagonal/total*100, 'r');
xlabel('Number of basic transformations');
ylabel('Representation accuracy (%)');
legend('proposed, alpha decreasing', 'proposed, alpha ones', 'Jacobi');

figure;
plot(diagonal_block/total*100, '-ms');
hold on; plot(diagonal_block_max/total*100, ':mv');
hold on; plot(diagonal_block_random/total*100, '--mp');
xlabel('Number of blocks');
ylabel('Representation accuracy (%)');
legend('proposed block', 'Jacobi block', 'random block');


%% get lowest eigenvalues via Algorithm 1
alpha = 1:p;
[positions, values, approx_error, tus, Ubar, S, diagonal] = algoritm1(-A, alpha, 10*g, p);

[positions_alpha_is_one, values_alpha_is_one, approx_error_alpha_is_one, tus_alpha_is_one, Ubar_alpha_is_one, S_alpha_is_one, diagonal_alpha_is_one] = ...
    algoritm1_alpha_is_ones(-A, 10*g, p);
	
%% plot results
total = -trace(L(1:p, 1:p));
figure;
plot(diagonal/total*100, 'k');
hold on; plot(diagonal_alpha_is_one/total*100, 'g');
