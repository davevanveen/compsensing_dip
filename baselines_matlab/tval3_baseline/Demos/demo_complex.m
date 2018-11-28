% demo_complex
% 
% x: dense vector (complex, one-dimentional)
% A: random matrix without normality and orthogonality (complex)
% f: observation with/without noise (complex)
%
% Due to the fact that A is totally random, high accuracy can still be 
% achieved by low recovery percentage for this demo.
%
% Written by: Chengbo Li
% Advisor: Prof. Yin Zhang and Wotao Yin
% CAAM department, Rice University
% 05/21/2009

clear; close all;
path(path,genpath(pwd));
fullscreen = get(0,'ScreenSize');

% problem size
n = 1000;
m = floor(.2*n);
k = floor(m/10);

% sensing matrix
A = (rand(m,n)-.5) + (rand(m,n)-.5)*1i;

% original staircase signal
xs = zeros(n,1);
p = randperm(n); p = sort(p(1:k-1)); p = [1 p n];
for sct = 1:k
    xs(p(sct):p(sct+1)) = rand + rand*1i;
end
nrmxs = norm(xs,'fro');

% observation
f = A*xs;
favg = mean(abs(f));

% add noise
sigma = 0.02;
f = f + sigma*favg*randn(m,1);

%% Run TVAL3
clear opts
opts.mu = 2^5;
opts.beta = 2^5;
opts.mu0 = 2^1;
opts.beta0 = 2^1;
opts.tol = 1E-4;
opts.maxit = 600;
opts.TVnorm = 1;

t = cputime;
[x, out] = TVAL3(A,f,n,1,opts);
t = cputime - t;
rerr = norm(x-xs,'fro')/nrmxs;

figure('Name','TVAL3','Position',...
    [fullscreen(1) fullscreen(2) fullscreen(3) fullscreen(4)]);
subplot(211); set(gca,'fontsize',16)
plot(1:n,real(xs),'ro',1:n,real(x),'b-');
title(sprintf('Real Part         Noise: %2.1f%%,   Rel-Err: %4.2f%%,   CPU: %4.2fs',sigma*100,rerr*100,t))
subplot(212); set(gca,'fontsize',16)
plot(1:n,imag(xs),'ro',1:n,imag(x),'b-');
title('Image Part                                                                             ')

plotting = 0;
if plotting
    figure(2);
    subplot(241); plot(out.lam1); title('\_al: ||w||');
    subplot(242); plot(out.lam2); title('\_al: ||Du-w||^2');
    subplot(243); plot(out.lam3); title('\_al: ||Au-f||^2');
    subplot(244); plot(abs(out.obj),'b-'); title('\_al: objective values');
    subplot(245); plot(out.res); title('\_al: residue');
    subplot(246); plot(abs(out.tau)); title('\_al: steplenths');
    subplot(247); plot(out.itrs); title('\_al: inner iterations');
    subplot(248); plot(abs(out.C),'r-'); title('\_al: reference vlaues');
    
    figure(3);
        semilogy(1:length(out.lam1),out.lam1,'b*:',1:length(out.lam2),sqrt(out.lam2),'rx:',...
        1:length(out.lam3),sqrt(out.lam3),'g.--', 1:length(out.f),sqrt(out.f),'m+-');
    legend('lam1(||w||_1)','lam2(||D(d_tu)-w||_2)','lam3(||Au-b||_2)','obj function');
end