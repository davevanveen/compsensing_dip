function Fpx = Fp1d(x,picks,n)

[p,q] = size(x);
msg = 'size of x not correct in Fp.m';

if q ~= 1 || p ~= n
    error(msg);
end

Fx = fft(x);
Fx = Fx(picks);
Fpx = [real(Fx); imag(Fx)];