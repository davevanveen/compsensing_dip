function FpTy = FpT1d(y,picks,n)

ly = length(y);
rP = length(picks);
if ly ~= 2*rP
    error('size not correct in FpT');
end

FpTy = zeros(n,1);
y = y(1:rP) + sqrt(-1)*y(rP+1:ly);
FpTy(picks) = y;
FpTy = real(ifft(FpTy));