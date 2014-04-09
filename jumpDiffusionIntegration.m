function [t,f,t_jump,f_jump_minus,f_jump] = jumpDiffusionIntegration(t0,f0,a,b,c,lambda,randJumpSize,T,N,M)
%% jumpDiffusionIntegration: Integration of SDE for diffusion with jumps
%
%% SYNTAX:
%        [t,f,t_jump,f_jump_minus,f_jump] = jumpDiffusionIntegration(t0,f0,a,b,c,lambda,randJumpSize,T,N,M)
%
%% INPUT:
%          t0 : Initial time
%          f0 : Initial value of the function f(t0) = f0;
%          a  : Handle of the function a(t,f) in the drift term
%          b  : Handle of the function b(t,f) in the diffusion term
%          c  : Handle of the function c(t,f) in the jump term
%     lambda  : Parameter of the Poisson counting process
% randJumpSize: Handle of the function that generates jump sizes
%          T  : Lenght of integration interval [t0, t0+T]
%          N  : Number of time steps
%          M  : Number of trajectories in simulation
%
%% OUTPUT:
%          t  : Times at which the trajectory is monitored
%               t(n) = t0 + n Delta T
%          f  : values of the trajectory that starts from
%                 f(1) = f0 at t0
%      t_jump : Times for jumps (cell array)
%f_jump_minus : Values of the trajectory right before jump
%      f_jump : Values of the trajectory right after jump
%
%% EXAMPLE:
%
%        %
%        %   Merton jump diffision model
%        %
%        % dS_t = mu S_t^- + sigma S_t^- dW_t + S_t^- dJ_t
%        %
%        S0 = 100; mu = 0.15; sigma = 0.2;
%        a = @(t,S)(mu*S);    % drift term in SDE
%        b = @(t,S)(sigma*S); % diffusion term in SDE
%        c = @(t,S)(S);       % jump term in SDE 
%        lambda = 5;  % Poisson process: average of lamdba jumps per year
%        randJumpSize = @()(lognrnd(0,0.2)-1); % random nr. generator for jump size
%        t0 = 1; T = 2;
%        N = 1000; M = 5;
%        [t,S,t_jump,S_jump_min,S_jump] = jumpDiffusionIntegration(t0,S0,...
%                  a,b,c,lambda,randJumpSize,T,N,M);
%        figure(1); plot(t,S)
%        hold on;
%        plot(t_jump{1},S_jump_min{1},'ro',t_jump{1},S_jump{1},'rx');
%        hold off;
%        xlabel('t'); ylabel('S(t)');
%        axis('tight');
%
%
deltaT = T/N;     % size of integration step
f = zeros(M,N+1); % initialize f
t = linspace(t0,t0+T,N+1); % initialize monitoring times
X = randn(M,N);   % generate X ~ N(0,1)
t_jump     = cell(M,1); % Cell array for vectors of jump times
f_jump_min = cell(M,1); % Cell array for vectors of value of f at t_jump^-
f_jump     = cell(M,1); % Cell array for vectors of value of f at t_jump
t(1) = t0;
f(:,1) = f0;
for m = 1:M
    
    %   Simulate the mth trajectory
    nJump  = 1;
    deltaT_jump = -log(rand)/lambda; % delay between consecutive jumps
    t_jump{m}(1) = t0 + deltaT_jump; % time of the first jump
    
    for n = 1:N
        tInit = t(n); 
        fInit = f(m,n);
        
        while(t(n+1) >= t_jump{m}(nJump)) % there is at least jump in [tInit,t_{n+1}]
            
            %  Integrate SDE in [tInit,t_jump)
            dT = t_jump{m}(nJump) - tInit; % size of interval
            f_jump_minus{m}(nJump) = fInit + a(tInit,fInit)*dT +...
                                             b(tInit,fInit)*sqrt(dT).*randn;
           
            %  Integrate SDE in (t_jump,t_jump]
            jumpSize  = randJumpSize(); % generate jump size
            f_jump{m}(nJump) = f_jump_minus{m}(nJump) +...
                jumpSize*c(t_jump{m}(nJump),f_jump_minus{m}(nJump));
            
            %  Generate time for new jump
            deltaT_jump = -log(rand)/lambda; % delay between consecutive jumps
            t_jump{m}(nJump+1) = t_jump{m}(nJump) + deltaT_jump;
            tInit = t_jump{m}(nJump);
            fInit = f_jump{m}(nJump);
            nJump = nJump + 1;
        end
        
        %  Integrate SDE in [tInit,t_{n+1}]
        dT = t(n+1)- tInit; % size of interval
        f(m,n+1) = fInit + a(tInit,fInit)*dT +...
                           b(tInit,fInit)*sqrt(dT).*X(m,n);
    end
    t_jump{m} = t_jump{m}(1:end-1);
end

