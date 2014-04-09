function [p,mu,sigma] = expectationMaximizationGM(Z,K,p,mu,sigma)
%% expectationMaximizationGM: EM algorithm for a mixture of K Gaussians GM[p,mu,sigma]
%
%% SYNTAX:
%         [p, mu, sigma] = expectationMaximizationGM(Z,K,p,mu,sigma)
%
%% INPUT:
%         Z : Sample with M observations 
%         K : Number of Gaussians in the mixture
%
%  OPTIONAL INPUT PARAMETERS (initial seed for EM)
%         p : vector of probabilitiess  [Kx1]
%        mu : vector of means           [Kx1]
%     sigma : vector of stdev's         [Kx1]
%
%% OUTPUT:
%         p : vector of probabilitiess  [Kx1]
%        mu : vector of means           [Kx1]
%     sigma : vector of stdev's         [Kx1]
%
%% EXAMPLE:
%
%   %% Sample matrix (M rows, N columns) 
%   M = 1000;
%   N = 1;
%   %% Gaussian components
%   p     = [ 1/2 1/3  1/6];  % probability vector 
%   mu    = [-1.0 4.0 12.0];  % means 
%   sigma = [ 1.0 3.0  0.5];  % standard deviations
%
%   %% Generate sample Z ~ GM[p,mu,sigma)
%   Z = GMrand(M,N,p,mu,sigma);
%
%   %% EM for Gaussian mixture
%   [p, mu, sigma] = expectationMaximizationGM(Z,3)
%
%   %% Compare modelPdf and scaled histogram
%   modelPdf = @(z)(GMpdf(z,p,mu,sigma));
%   figure(1); graphicalComparisonPdf(Z(:),modelPdf)
%   title('Z ~ GM[p,\mu,\sigma]')
%
if(K==1)
    %
    %  Fit to a single Gaussian
    %   
    p=1; mu = mean(Z); sigma = std(Z);
    return
end
%
if(nargin == 2)
    %
    %  Automatic initialization of parameters
    %
    mu0 = mean(Z);
    sigma0 = std(Z);
    mu = linspace( mu0 - 2*sigma0, mu0 + 2*sigma0,K);
    sigma = 2*sigma0*ones(size(mu));
    p = ones(1,K)/K;
end
Z = Z(:)';      % Ensures that Z is a row vector
M = length(Z);  % Size of the sample
%
%
a = 0.0; b = 0.0; c = 0.0; m = 0.0;   % Standard EM
% a = 0.2; b = 0.2; c = 0.1; m = 0.0; % Modified EM (Hamilton, 1991)
%
TOL_MU = 1e-6; TOL_SIGMA = 1e-6; TOL_P  = 1e-6;
MAXITER = 5000;

%%  EM Algotrithm
%  
dmu = 10*TOL_MU; dsigma = 10*TOL_SIGMA; dp = 10*TOL_P;
iter = 0;
while(iter < MAXITER && (any(abs(dmu) >  TOL_MU) || any(abs(dsigma) > TOL_SIGMA) || any (abs(dp) > TOL_P)))
    iter = iter + 1;    % increment the iteration counter
    norm = zeros(1,M);  
    for j = 1:K
        gamma(j,:) = p(j)*normpdf(Z,mu(j),sigma(j));
        norm = norm + gamma(j,:);
    end
    %
    dp = -p;
    dmu = -mu;
    dsigma = -sigma;
    %
    for j = 1: K
        gamma(j,:)  =  gamma(j,:)./norm;
        norm2 = sum(gamma(j,:));
        p(j) = norm2/M;
        mu(j) = (c*m + sum(gamma(j,:).*Z))/(c + norm2);
        sigma(j) = sqrt((b + c*(m-mu(j))^2 + ...
            sum(gamma(j,:).*(Z-mu(j)).^2))/(a + norm2));
    end
    %
    dp = dp + p;
    dsigma = dsigma + sigma;
    dmu = dmu + mu;
    %
end
if(iter == MAXITER)
    warning('Warning: EM for GM has not fully converged. Try a smaller number of Gaussians',gcf);
end
