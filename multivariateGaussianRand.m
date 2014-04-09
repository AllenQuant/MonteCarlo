function Z = multivariateGaussianRand(M,mu,Sigma)
%% multivariateGaussianRand: Generate random numbers from a D-dimensional Gaussian
%
%% SYNTAX:
%         Z = multivariateGaussianrand(M,mu,Sigma)
%
%% INPUT:
%         M : size of the sample
%        mu : vector of means   [D,1]
%     Sigma : covariance matrix [D,D]
%
%% OUTPUT:
%         Z : Sample from N(mu,Sigma) Gaussian  [M,D]  
%
%% EXAMPLE:   
%        mu = [1;3]; 
%        Sigma = [1 0.8; 0.8 2];
%        M = 1000;
%        Z = multivariateGaussianRand(M,mu,Sigma);
%        figure(1);
%        graph2DGaussian(mu,Sigma);
%        hold on;
%        plot(Z(:,1),Z(:,2),'.');
%        hold off;

%% Dimension of the multivariate normal
D = length(mu);

%% Cholesky decomposition of the covariance matrix
L = chol(Sigma)';

%% Generate X ~ N(0,I) in D dimensions   (independent components)
X = randn(D,M);

%% Generate Z ~ N(mu,Sigma) in D dimensions (linearly dependent components) 
Z = repmat(mu,1,M) + L*X;  % Matrix [D,M]
Z = Z';                    % Matrix [M,D]
