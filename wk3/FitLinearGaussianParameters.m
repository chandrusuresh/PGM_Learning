function [Beta sigma] = FitLinearGaussianParameters(X, U)

% Estimate parameters of the linear Gaussian model:
% X|U ~ N(Beta(1)*U(1) + ... + Beta(n)*U(n) + Beta(n+1), sigma^2);

% Note that Matlab/Octave index from 1, we can't write Beta(0).
% So Beta(n+1) is essentially Beta(0) in the text book.

% X: (M x 1), the child variable, M examples
% U: (M x N), N parent variables, M examples
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

M = size(U,1);
N = size(U,2);

Beta = zeros(N+1,1);
sigma = 1;

% collect expectations and solve the linear system
% A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ; 
%       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
%       ...         , ...         , ... , ...         , ...   ;
%       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]

% construct A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1 = zeros(N,N);
A2 = zeros(1,N);
for i = 1:N
    A1(i,i) = mean(U(:,i).*U(:,i));
    A2(1,i) = mean(U(:,i));
    for j = i+1:N
        A1(j,i) = mean(U(:,j).*U(:,i));
        A1(i,j) = A1(j,i);
    end
end
A = [A2,1;A1,A2'];

% B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]
% construct B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = zeros(N+1,1);
B(1) = mean(X);
for i = 2:length(B)
    B(i) = mean(X.*U(:,i-1));
end

% solve A*Beta = B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Beta = A\B;

% then compute sigma according to eq. (11) in PA description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma = (std(X,1))^2;
for i = 1:N
    for j = 1:N
        cov_Ui_Uj = A1(i,j) - A2(i)*A2(j);
        sigma = sigma - Beta(i)*Beta(j)*cov_Ui_Uj;
    end
end
sigma = sqrt(sigma);
