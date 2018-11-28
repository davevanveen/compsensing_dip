% Demonstrates compressively sampling and TVAL3 recovery of an image.

% NOTE: Need to fix absolute paths, add user-specified arguments
%      i.e. code not currently user-friendly 

clear all; close all;
path(path,genpath(pwd));
data_path = '/home/dave/Documents/comp_sensing/cs_dip/csdip_code/cs_dip/icml_push/data/xray/sub';
tval3_path = '/home/dave/Documents/comp_sensing/cs_dip/csdip_code/cs_dip/TVAL3_v1.0';
addpath(genpath(data_path));
addpath(genpath(tval3_path));
cd(data_path)
imagefiles = dir('*.jpg');

cd(tval3_path)

% params to change
num_images = 60;
imsize=256;
%num_meas_list = [10, 15, 25, 35, 50, 75, 100, 200]; %mnist
num_meas_list = [500, 1000, 1500, 2000, 4000, 8000]; %xray

num_pixels = imsize^2;
samp_rate_list = num_meas_list ./ num_pixels;

% Set TVAL3 params
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
    fn_out = strcat('im', num2str(num), '.mat')
    
    x_0=imresize(ImIn,imsize/size(ImIn,1));
    [height, width]=size(x_0);
    n=length(x_0(:));

    for jj=1:length(num_meas_list)
        path_data_out = strcat('~/Documents/comp_sensing/cs_dip/csdip_code/cs_dip', ...
        '/icml_push/reconstructions/xray/tval3/meas', num2str(num_meas_list(jj)), sprintf('/'));
        cd(path_data_out); 

        SamplingRate = samp_rate_list(jj);
        m = round(n*SamplingRate);
        M=randn(m,n); % generate random gaussian meas. matrix
        for j = 1:n
            M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
        end
        y = M*x_0(:);
        [x_hat,out] = TVAL3(M,y,imsize,imsize,opts);

        save(fn_out, 'x_hat');
    end
end