function [A, At] = getA(m,p,q,picks,mtx,dim)

if strcmp(mtx,'randn')
    B = randn(m,p*q);

    % random A -- 1
    [B,R] = qr(B',0);
    B = B';
    
    %     %% random A -- 2
    %     d = sqrt(sum(B'.^2));
    %     B = sparse(1:m,1:m,1./d)*B;


    A = @(x) B*x;
    At = @(y) (y'*B)';
elseif strcmp(mtx,'PF')
    if dim == 2
        % 2D partial Fourier
        A = @(x) Fp2d(x,picks,p,q);
        At = @(y) FpT2d(y,picks,p,q);
    elseif dim == 1
        % 1D partial Fourier
        A = @(x) Fp1d(x,picks,p);
        At = @(y) FpT1d(y,picks,p);
    end
end