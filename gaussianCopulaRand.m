function U = gaussianCopulaRand(M,rho);
%% gaussianCopulaRand: Generates random numbers from a Gaussian copula
%
%% SYNTAX:
%         U = gaussianCopulaRand(M,rho)
%
%% INPUT:
%         M : size of the sample
%       rho : corelation matrix [D,D]
%
%% OUTPUT:
%         U : Sample from the copula [M,D]  
%
%% EXAMPLE:   
%        rho = [1 0.8; 0.8 1];
%        M = 1e3;
%        U = gaussianCopulaRand(M,rho);
%        figure(1);
%        graph2DGaussian(zeros(2,1),rho);
%        hold on;
%        plot(norminv(U(1,:)),norminv(U(2,:)),'.');
%        hold off;

%% Dimensionality of the copula
D = size(rho,1); 

%% Cholesky decomposition
L = chol(rho)';

%% Sample from N(0,I)   [independent components]
X = randn(D,M);     

%% Sample from N(0,rho) [linearly dependent components]
Z = L*X;                           

%% Gaussian copula (U[0,1] marginals)
U = normcdf(Z);  % Matrix [D,M]
U = U';          % Matrix [M,D]     
