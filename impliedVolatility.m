function sigma = impliedVolatility(fPrice,fVega,price)
% impliedVolatility: Implied volatility of a derivative
%
% SYNTAX:
%       sigma = impliedVolatility(fPrice,fVega,price)
%
% INPUT:
%    fPrice : Handle to the function that gives the price of the derivative
%     fVega : Handle to the function that gives the vega of the derivative
%     price : The price of the derivative
%
% OUTPUT:
%     sigma : Implied volatility
%
% EXAMPLE:   
%        S0 = 100; K = 90; r = 0.05; T = 2;
%        price = 19.8701;
%        fPrice = @(sigma)(priceEuropeanCall(S0,K,r,T,sigma));
%        fVega  = @(sigma)(vegaEuropeanCall(S0,K,r,T,sigma));
%        impliedSigma  = impliedVolatility(fPrice,fVega,price)
%        priceEuropeanCall(S0,K,r,T,impliedSigma)  % should be equal to price           
%          

TOLABS  = 1e-6;
MAXITER = 100;

sigma  = 0.3;         % intial estimate   
dSigma = 10*TOLABS;   % enter loop for the first time
nIter  = 0;
while (nIter < MAXITER && abs(dSigma) > TOLABS)
    nIter = nIter + 1;
    % Newton-Raphson method
    dSigma = (fPrice(sigma)-price)/fVega(sigma); 
    sigma = sigma - dSigma;  
end

if (nIter == MAXITER)
    warning('Newton-Raphson not converged')
end
