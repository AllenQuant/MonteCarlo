function [price_MC,stdev_MC] = priceEuropeanCallMC_importanceSampling(S0,K,r,T,sigma,M)
%% priceEuropeanCallMC_importanceSampling: Price of a European call option in the Black-Scholes model with importance sampling
%
%% SYNTAX:
%        [price_MC,stdev_MC] = priceEuropeanCallMC_importanceSampling(S0,K,r,T,sigma,M)
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
%        S0 = 100; K = 200; r = 0.05; T = 2; sigma = 0.4;
%        M = 1e4; 
%        [price_MC,stdev_MC] = priceEuropeanCallMC(S0,K,r,T,sigma,M)
%        [price_MC_importanceSampling,stdev_MC_importanceSampling] = priceEuropeanCallMC_importanceSampling(S0,K,r,T,sigma,M)
%          

%% Generate M samples from N(0,1)
X = randn(M,1); 

%% Simulate M trajectories in one step

mu = log(K/S0)/T;
ST = S0*exp((mu-0.5*sigma^2)*T + sigma*sqrt(T)*X);
correctionFactor = normpdf(X + (mu-r)*sqrt(T)/sigma)./normpdf(X);

%% Compute the payoffs
payoff  = max(ST-K,0);
correctedPayoff = payoff.*correctionFactor;

%% MC estimate of the price and the error of the option
discountFactor = exp(-r*T);

price_MC = discountFactor*mean(correctedPayoff);
stdev_MC = discountFactor*std(correctedPayoff)/sqrt(M);
