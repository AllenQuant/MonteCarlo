function U = studentTCopulaRand(M,rho,nu);
%% studentTCopulaRand: Generates random numbers from a Student's t copula
%
%% SYNTAX:
%         U = studentTCopulaRand(M,rho,nu)
%
%% INPUT:
%         M : size of the sample
%       rho : corelation matrix [D,D]
%        nu : degrees of freedom
%
%% OUTPUT:
%         U : Sample from the copula [M,D]  
%
%% EXAMPLE:   
%        rho = [1 0.8; 0.8 1];
%        M = 1e4;
%        U = studentTCopulaRand(M,rho);
%        figure(1);
%        plot(U(1,:),U(2,:),'.');

%% Dimensionality of the copula
D = length(rho); 

%% Cholesky decomposition
L = chol(rho)';

%% Sample from N(0,I)   [independent components]
X = randn(D,M);     

%% Sample from N(0,rho) [linearly dependent components]
Z = L*X;                           

%% Sample from Chi2 with nu degrees of freedom
xi = chi2rnd(nu,1,M);
xi = repmat(xi,D,1);

%% Student t copula (U[0,1] marginals)
U  = tcdf(sqrt(nu)*Z./sqrt(xi),nu); % Matrix [D,M]
U = U';                             % Matrix [M,D]

