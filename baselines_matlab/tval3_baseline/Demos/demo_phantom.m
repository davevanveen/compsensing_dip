% demo_phantom
%
% This demo shows how TVAL3 handles complex measurement matrix and how to
% trigger the continuation scheme to speed up the convergence.
% 
% I: 64x64 phantom (real, two-dimentional)
% A: random complex matrix without normality and orthogonality (complex)
% f: observation with/without noise (real)
%
% Written by: Chengbo Li
% Advisor: Prof. Yin Zhang and Wotao Yin
% CAAM department, Rice University
% 05/21/2009

clear; close all;
path(path,genpath(pwd));
fullscreen = get(0,'ScreenSize');

% problem size
n = 64;
ratio = .3;
p = n; q = n; % p x q is the size of image
m = round(ratio*n^2);

% sensing matrix
% A = randn(m,p*q) + i*randn(m,p*q);
A = (rand(m,p*q)-.5) + 1i*(rand(m,p*q)-.5);

% original image
I = phantom(n);
nrmI = norm(I,'fro');

% observation
sigma = .00;
f = A*I(:);
favg = mean(abs(f));
f = f + sigma*favg*(randn(m,1) + 1i*randn(m,1));

figure('Name','TVAL3','Position',...
    [fullscreen(1) fullscreen(2) fullscreen(3) fullscreen(4)]);
subplot(121); imshow(I,[]);
title('Original','fontsize',18); drawnow;
xlabel(sprintf('Noise level: %2d%%',sigma*100),'fontsize',14);

%% Run TVAL3
clear opts
opts.mu = 2^13;
opts.beta = 2^9;
opts.mu0 = 2^4;      % trigger continuation shceme
opts.beta0 = 2^0;    % trigger continuation scheme
opts.tol_inn = 1e-4;
opts.tol = 1e-5;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;
%opts.disp = true;
opts.Ut = I;


t = cputime;
[U, out] = TVAL3(A,f,p,q,opts);
t = cputime - t;

subplot(122); 
imshow(real(U),[]);
title('Recovered by TVAL3','fontsize',18);
xlabel(sprintf(' %2d%% measurements \n Rel-Err: %5.4f%%,   CPU: %4.2fs ',...
    ratio*100,norm(U-I,'fro')/nrmI*100,t),'fontsize',16);


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