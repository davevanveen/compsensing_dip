%
% set paths: users must make sure that all folders and subfolders
% are in cluded in MATLAB working paths. This can be done either 
% manually or by running the following line in MATLAB command window
%
path(path,genpath(pwd));
if exist('fWHtrans','file') ~= 3
    cd Fast_Walsh_Hadamard_Transform;
    mex -O fWHtrans.cpp
    cd ..;
    fprintf('Finished compiling the C++ code for fast Walsh-Hadamard transform!\n');
end
fprintf('Finished adding paths! Welcome to use TVAL3 version 1.0.\n');