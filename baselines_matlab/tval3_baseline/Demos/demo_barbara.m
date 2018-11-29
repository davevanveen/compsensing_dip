function demo_barbara
%
% This demo shows how TVAL3 handles a complicated image and how to trigger
% the continuation scheme in order to speed up the convergence.
%
% I: 256x256 Barbara (real, two-dimentional)
% A: permuted Walsh Hadamard transform (real)
% f: observation with noise (real)
% 
% Written by: Chengbo Li
% Advisor: Prof. Yin Zhang and Wotao Yin 
% CAAM department, Rice University
% 05/21/2009

clear all; close all;
path(path,genpath(pwd));
fullscreen = get(0,'ScreenSize');

% problem size
ratio = .6;
sidelength = 256;
N = sidelength^2;
M = round(ratio*N);

% original image
bbr = importdata('barbara256.tif');
Im = double(bbr(:,:,3));

% generate measurement matrix
p = randperm(N);
picks = p(1:M);
for ii = 1:M
    if picks(ii) == 1
        picks(ii) = p(M+1);
        break;
    end
end
perm = randperm(N); % column permutations allowable
A = @(x,mode) dfA(x,picks,perm,mode);

% observation
b = A(Im(:),1);
bavg = mean(abs(b));

% add noise
sigma = 0.04;  % noise std
noise = randn(M,1);
b = b + sigma*bavg*noise;

% set the optional paramaters
clear opts
opts.mu = 2^12;
opts.beta = 2^6;
opts.mu0 = 2^4;       % trigger continuation shceme
opts.beta0 = 2^-2;    % trigger continuation shceme
opts.maxcnt = 10;
opts.tol_inn = 1e-3;
opts.tol = 1E-6;
opts.maxit = 300;

% reconstruction
t = cputime;
[estIm, out] = TVAL3(A,b,sidelength,sidelength,opts);
estIm = estIm - min(estIm(:));
t = cputime - t;
re_er = norm(estIm-Im,'fro')/norm(Im,'fro');


% plotting
figure('Name','Barbara','Position',...
    [fullscreen(1) fullscreen(2) fullscreen(3) fullscreen(4)]);
colormap(gray);

subplot(1,2,1);
imshow(Im,[]);
title(sprintf('Original %dx%d Barbara',sidelength,sidelength),'fontsize',16);
subplot(1,2,2);
imshow(estIm,[]);
title(sprintf('Recovered Barbara with %2.0f%% measurements',ratio*100),'fontsize',16);
xlabel(sprintf('Noise level: %2d%%  \n Rel-Err: %4.2f%%,   CPU time: %4.2fs',100*sigma,100*re_er,t),'fontsize',14);

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% dfA
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = dfA(x,picks,perm,mode)
switch mode
    case 1
        y = A_fWH(x,picks,perm);
    case 2
        y = At_fWH(x,picks,perm);
    otherwise
        error('Unknown mode passed to f_handleA!');
end
