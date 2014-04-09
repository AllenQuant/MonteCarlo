function graph2DGaussian(mu,Sigma,alpha,nContours)
%% graph2DGaussian: contour plot and 3D-graph of a 2-dimensional Gaussian
%
%% SYNTAX:
%         h = graph2DGaussian(mu,Sigma,alpha,nContours)
%
%% INPUT:
%        mu : Vector of means   [2x1]
%     Sigma : Covariance matrix [2x2]
%     alpha : Extent of the plotting area           [default = 3]
% nContours : Number of contour curves in the plot  [default = 25] 
%
%% OUTPUT:
%        None  
%
%% EXAMPLE:   
%        mu = [1;3]; 
%        Sigma = [1 0.8; 0.8 2];
%        graph2DGaussian(mu,Sigma);
%

%% Default options
if(nargin < 4)
    nContours = 25;
    if(nargin < 3)
        alpha = 3;
    end
end

%% Generate 2-dimensional grid
nPlot = 100;

radius1 = alpha*sqrt(Sigma(1,1));
radius2 = alpha*sqrt(Sigma(2,2));

z1 = linspace(mu(1)-radius1,mu(1)+radius1,nPlot);
z2 = linspace(mu(2)-radius2,mu(2)+radius2,nPlot);

[Z1,Z2] = meshgrid(z1,z2);

%% Compute the pdf on the grid
detSigma = det(Sigma);
invSigma = inv(Sigma);
[nRows,nColumns] = size(Z1);
for i = 1:nRows
    for j = 1:nColumns
          z = [Z1(i,j); Z2(i,j)]- mu;
          f(i,j) = exp(-z'*invSigma*z/2.0)/(2.0*pi*sqrt(detSigma));
    end
end

%% Graph in 3 dimensiones
figure(1);
mesh(Z1,Z2,f)
axis('tight')

%% Contour plot in 2 dimensions
figure(2);
contour(Z1,Z2,f,nContours);
axis('tight')
