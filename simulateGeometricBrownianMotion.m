function [t,S] = simulateGeometricBrownianMotion(M,N,t0,S0,T,mu,sigma)
%% simulateGeometricBrownianMotion: Simulate geometric Brownian motion in [t0,t0+T]
%
%% SYNTAX:
%        [t,S] = simulateGeometricBrownianMotion(M,N,t0,S0,T,mu,sigma)
%
%% INPUT:
%        M  : Number of trajectories
%        N  : Number of time steps
%        t0 : initial time
%        S0 : initial value S(t0) = S0
%        T  : Length of the simulation interval [t0,t0+T]
%  mu,sigma : parameters of the process
%
%% OUTPUT:
%        t  : Times at which the trajectory is monitored
%        S  : Simulation of M trajectories of geometric Brownian motion
%
%% EXAMPLE:   
%
%       t0    = 1;    % initial time
%       S0    = 100;  % initial value of the underlying
%       T     = 2;    % interval [t0,t0+T]  
%       N     = 100;  % number of time steps   
%       M     = 50;   % number of trajectories 
%       mu    = 0.4;  % expecter return   
%       sigma = 0.3;  % volatility
%       [t,S] = simulateGeometricBrownianMotion(M,N,t0,S0,T,mu,sigma);
%       figure(1);  clf
%       plot(t,S')
%       xlabel('t'); ylabel('S(t)');
%

%% Size of the integration step
deltaT = T/N;          

%% Monitoring times
t = linspace(t0,t0+T,N+1); 

%% Simulate M trajectories 
X = randn(M,N);  % Gaussian white noise
e = exp((mu-0.5*sigma^2)*deltaT + sigma*sqrt(deltaT)*X);
S = cumprod([S0*ones(M,1) e], 2);
