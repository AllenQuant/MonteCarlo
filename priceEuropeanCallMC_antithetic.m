function [price_MC,stdev_MC] = priceEuropeanCallMC_antithetic(S0,K,r,T,sigma,M)
%% priceEuropeanCallMC_antithetic: Price of a European call option in the Black-Scholes model with antithetic variates
%
%% SYNTAX:
%        [price_MC,stdev_MC] = priceEuropeanCallMC_antithetic(S0,K,r,T,sigma,M)
%
%% INPUT:
%       S0 : Initial value of the underlying asset
%        K : Strike 
%        r : Risk-free interest rate 
%        T : Time to expiry 
%    sigma : Volatility 
%        M : Number of simulations
% 
%
%% OUTPUT:
% price_MC : MC estimate of the price of the option in the Black-Scholes model  
% stdev_MC : MC estimate of the standard deviation  
%
%% EXAMPLE:   
%        S0 = 100; K = 90; r = 0.05; T = 2; sigma = 0.4;
%        M = 1e4; 
%        [price_MC,stdev_MC] = priceEuropeanCallMC(S0,K,r,T,sigma,M)
%        [price_MC_antithetic,stdev_MC_antithetic] = priceEuropeanCallMC_antithetic(S0,K,r,T,sigma,M)
%          

%% Generate M samples from N(0,1)
X = randn(M,1); 

%% Simulate M trajectories in one step

ST_plus  = S0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*X);
ST_minus = S0*exp((r-0.5*sigma^2)*T - sigma*sqrt(T)*X);

%% Compute payoffs
payoff_plus  = max(ST_plus-K,0);
payoff_minus = max(ST_minus-K,0);

% average payiff
payoff       = 0.5*(payoff_plus+payoff_minus);

%% MC estimate of the price and the error of the option
discountFactor = exp(-r*T);
price_MC = discountFactor*mean(payoff);
stdev_MC = discountFactor*std(payoff)/sqrt(M);
