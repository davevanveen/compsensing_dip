% Demonstrates compressively sampling and TVAL3 recovery of an image.
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOTE: Currently this code requires user setting variables
%       and ensuring the current path is correct.
%       All user input is contained within this section.

% current path should be ../baselines_matlab/tval3_baseline
base_direc = pwd;

%Parameters - user must ensure these are correct
dataset = 'xray'; % 'xray' or 'mnist'
imsize = 256; % imsize: {mnist: 28, xray: 256}

%num_meas_list = [25, 35, 50, 75, 100, 200]; %mnist
num_meas_list = [500,1000,2000,4000,8000]; %xray
num_images = 1; % number of images to reconstruct
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_pixels = imsize^2;
samp_rate_list = num_meas_list ./ num_pixels;

path_data_in = '../../data/xray/sub'; 
cd(path_data_in)
imagefiles = dir('*.jpg');
cd(base_direc)

% TVAL3 optimization params
clear opts
opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;

for ii=1:num_images
    % Data I/0
    fn_in = imagefiles(ii).name;
    ImIn = double(imread(fn_in));
    num = sscanf(fn_in, '%d.jpg');
    fn_out = strcat('im', num2str(num), '.mat');
    
    x_0=imresize(ImIn,imsize/size(ImIn,1));
    [height, width]=size(x_0);
    n=length(x_0(:));

    for jj=1:length(num_meas_list)
        path_data_out = strcat('../../reconstructions/',dataset,'/tval3/meas', ... 
            num2str(num_meas_list(jj)), sprintf('/'));
        
        SamplingRate = samp_rate_list(jj);
        m = round(n*SamplingRate);
        M=randn(m,n); % generate random gaussian meas. matrix
        for j = 1:n
            M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
        end
        y = M*x_0(:);
        [x_hat,out] = TVAL3(M,y,imsize,imsize,opts);
        
        cd(path_data_out); 
        save(fn_out, 'x_hat');
        cd (base_direc);
    end
end