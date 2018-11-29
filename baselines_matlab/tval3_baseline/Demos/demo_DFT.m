function demo_DFT
% 
% x: sparse vector (complex, one-dimentional)
% A: permuted discrete Fourier transform (complex)
% f: observation with/without noise (complex)
%
%
% Written by: Chengbo Li
% Advisor: Prof. Yin Zhang and Wotao Yin
% CAAM department, Rice University
% 05/21/2009

clear all; close all;
path(path,genpath(pwd));
fullscreen = get(0,'ScreenSize');

% problem size
n = 2^14;
m = floor(.25*n); k = floor(.01*m);
fprintf('\nSize [n,m,k] = [%i,%i,%i]\n',n,m,k);

% generate random xs
% xs = zeros(n,1);
% p = randperm(n); p = sort(p(1:k-1)); p = [1 p n];
% xs(p(1:k)) = randn(k,1) + randn(k,1)*1i;
    
% generate staircase xs
z = zeros(n-1,1);
p = randperm(n-1);
z(p(1:k)) = randn(k,1) + 1i*randn(k,1);
xs = cumsum([randn; z]);



% generate partial DFT data
p = randperm(n);
picks = sort(p(1:m)); picks(1) = 1;
perm = randperm(n); % column permutations allowable
A = @(x,mode) dfA(x,picks,perm,mode);
b = A(xs,1); 
bavg = mean(abs(b)); 

% add noise
sigma = 0.05;  % noise std
noise = randn(m,1) + 1i*randn(m,1);
b = b + sigma*bavg*noise;

% set solver options
opts.maxit = 600;
opts.mu = 2^4;
opts.beta = 2^6;
opts.tol = 1E-4;
opts.TVnorm = 1;
opts.TVL2 = true;

t = cputime;
[x,out] = TVAL3(A, b, n, 1, opts);
t = cputime - t;

rerr = norm(x-xs)/norm(xs);


figure('Name','TVAL3','Position',...
    [fullscreen(1) fullscreen(2) fullscreen(3) fullscreen(4)]);
subplot(211); set(gca,'fontsize',16)
plot(1:n,real(xs),'r.-',1:n,real(x),'b-');
title(sprintf('Real Part         Noise: %2.1f%%,   Rel-Err: %4.2f%%,   CPU: %4.2fs',sigma*100,rerr*100,t))
subplot(212); set(gca,'fontsize',16)
plot(1:n,imag(xs),'r.-',1:n,imag(x),'b-');
title('Image Part                                                                             ')

plotting = 0;
if plotting
    figure(2);
    subplot(241); plot(out.lam1); title('\_al: ||w||');
    subplot(242); plot(out.lam2); title('\_al: ||Du-w||^2');
    subplot(243); plot(out.lam3); title('\_al: ||Au-f||^2');
    subplot(244); plot(abs(out.obj),'r-'); title('\_al: objective values');
    subplot(245); plot(out.res); title('\_al: residue');
    subplot(246); plot(abs(out.tau)); title('\_al: steplenths');
    subplot(247); plot(out.itrs); title('\_al: inner iterations');
    subplot(248); plot(abs(out.C),'r-'); title('\_al: reference vlaues');

    figure(3);
        semilogy(1:length(out.lam1),out.lam1,'b*:',1:length(out.lam2),sqrt(out.lam2),'rx:',...
        1:length(out.lam3),sqrt(out.lam3),'g.--', 1:length(out.f),sqrt(out.f),'m+-');
    legend('lam1(||w||_1)','lam2(||D(d_tu)-w||_2)','lam3(||Au-b||_2)','obj function');
end



    
 function y = dfA(x,picks,perm,mode)

A = pdft_operator(picks,perm);
switch mode
    case 1
        % y = A_fw(z, OMEGA, idx, permx);
        y = A.times(x);
    case 2
        % y = At_fw(z, OMEGA, idx, permx);
        y = A.trans(x);
    otherwise
        error('Unknown mode passed to f_handleA in ftv_cs.m');
end   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = pdft_operator(picks,perm)

% Define A*x and A'*y for a partial DFT matrix A
% Input:
%            n = interger > 0
%        picks = sub-vector of a permutation of 1:n
% Output:
%        A = struct of 2 fields
%            1) A.times: A*x
%            2) A.trans: A'*y
A.times = @(x) pdft_n2m(x,picks,perm);
A.trans = @(y) pdft_m2n(y,picks,perm);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = pdft_n2m(x,picks,perm)

% Calculate y = A*x,
% where A is m x n, and consists of m rows of the 
% n by n discrete-Fourier transform (FFT) matrix.
% The row indices are stored in picks.

x = x(:);
n = length(x);
tx = fft(x(perm))/sqrt(n);
y = tx(picks);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = pdft_m2n(y,picks,perm)

% Calculate x = A'*y,
% where A is m x n, and consists of m rows of the 
% n by n inverse discrete-Fourier transform (IFFT) 
% matrix. The row indices are stored in picks.

n = length(perm); 
tx = zeros(n,1);
tx(picks) = y;
x = zeros(n,1);
x(perm) = ifft(tx)*sqrt(n);

