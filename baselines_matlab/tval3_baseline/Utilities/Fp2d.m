function Fpx = Fp2d(x,picks,m,n)

[p,q] = size(x);
msg = 'size of x not correct in Fp2d.m';

if q == 1
    if p ~= (m*n)
        error(msg);
    else
        x = reshape(x,[m,n]);
    end
elseif p ~= m || q ~= n
        error(msg);   
end

Fx = fft2(x)/sqrt(m*n);
Fx = Fx(picks);
Fpx = [real(Fx); imag(Fx)];
