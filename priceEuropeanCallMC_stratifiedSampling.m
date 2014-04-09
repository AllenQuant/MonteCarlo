function [price_MC,stdev_MC] = priceEuropeanCallMC_stratifiedSampling(S0,K,r,T,sigma,M,nStrata)
%% priceEuropeanCallMC_stratifiedSampling: Price of a European call option in the Black-Scholes model with stratified sampling
%
%% SYNTAX:
%        [price_MC,stdev_MC] = priceEuropeanCallMC_stratifiedSampling(S0,K,r,T,sigma,M,nStrata)
%
%% INPUT:
%       S0 : Initial value of the underlying asset
%        K : Strike 
%        r : Risk-free interest rate 
%        T : Time to expiry 
%    sigma : Volatility 
%        M : Number of simulations
%  nStrata : Number of strata per simulation 
%
%% OUTPUT:
% price_MC : MC estimate of the price of the option in the Black-Scholes model  
% stdev_MC : MC estimate of the standard deviation  
%
%% EXAMPLE:   
%        S0 = 100; K = 90; r = 0.05; T = 2; sigma = 0.4;
%        M = 1e4; 
%        [price_MC,stdev_MC] = priceEuropeanCallMC(S0,K,r,T,sigma,M)
%        nStrata = 10; M = round(M/nStrata);
%        [price_MC_stratifiedSampling,stdev_MC_stratifiedSampling] = priceEuropeanCallMC_stratifiedSampling(S0,K,r,T,sigma,M,nStrata)
%          

%% Generate M x nStrata samples from U(0,1)
U = rand(M,nStrata);

% delimit the strata
strataBorders = linspace(0,1,nStrata+1);
strataSize    = strataBorders(2)-strataBorders(1);

% stratify U[0,1]
V = repmat(strataBorders(1:end-1),M,1) + U*strataSize; 

% stratified N(0,1)
X = norminv(V);

%% Simulate M trajectories in one step
ST  = S0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*X);

%% Compute the payoffs
payoff  = mean(max(ST-K,0),2); % average payoff for each set of strata

%% MC estimate of the price and the error of the option
discountFactor = exp(-r*T);

price_MC = discountFactor*mean(payoff);
stdev_MC = discountFactor*std(payoff)/sqrt(M);
