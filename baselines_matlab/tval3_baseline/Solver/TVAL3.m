function [U,out] = TVAL3(A,b,p,q,opts)
% Accordingly choose the code based on the model selected by the user.
%
% 1) TV model:        min sum ||D_i u||. 
%                        s.t. Au = b
% 2) TV/L2 model:     min sum ||D_i u|| + mu/2||Au-b||_2^2 
%
% Please use the default one if the user doesn't have a specific model to
% solver.
%
% Written by: Chengbo Li
% Advisor: Prof. Yin Zhang and Wotao Yin
% Computational and Applied Mathematics department, Rice University
% May. 15, 2009

if ~isfield(opts,'TVL2')
    opts.TVL2 = false;
end

if opts.TVL2
     [U, out] = ftvcs_al_TVL2p(A,b,p,q,opts);
else 
    [U, out] = ftvcs_alp(A,b,p,q,opts);
end