function res = isinInterval(val,lb,ub,eok)

% Checks to make sure that val is a real scalar in [lb,ub] if eok is true, 
% or in (lb,ub) if eok is false.

res = true;

if ~isscalar(val), res = false; end
if ~isreal(val), res = false; end
if eok
    if val < lb || val > ub, res = false; end
else
    if val <= lb || val >= ub, res = false; end
end

return