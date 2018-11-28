function FpTy = FpT2d(y,picks,m,n)

ly = length(y);
rP = length(picks);
if ly ~= 2*rP
    error('size not correct in FpT2d');
end

FpTy = zeros(m,n);
y = y(1:rP) + sqrt(-1)*y(rP+1:ly);
FpTy(picks) = y;
FpTy = real(ifft2(FpTy)*sqrt(m*n));
FpTy = FpTy(:);