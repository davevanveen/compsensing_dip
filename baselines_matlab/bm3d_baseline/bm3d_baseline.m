% Demonstrates compressively sampling and D-AMP recovery of an image.
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOTE: Currently this code requires user setting variables
%       and ensuring the current path is correct.
%       All user input is contained within this section.

% current path should be ../baselines_matlab/bm3d_baseline
base_direc = pwd;

%Parameters - user must ensure these are correct
dataset = 'xray'; % 'xray', 'mnist', or 'retino'
denoiser1 = 'BM3D'; % BM3D if grayscale or CBM3D if rgb
rgb = false; % false if grayscale, true if rgb
imsize = 256; % imsize: {mnist: 28, xray: 256, retino: 128}

%num_meas_list = [25, 35, 50, 75, 100, 200]; %mnist
num_meas_list = [500,1000,2000,4000,8000]; %xray or retino
num_images = 1; % number of images to reconstruct
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_pixels = imsize^2;
samp_rate_list = num_meas_list ./ num_pixels;

path_bm3d = './BM3D'; addpath(genpath(path_bm3d));
path_utils = './Utils'; addpath(genpath(path_utils));
path_data_in = '../../data/xray/sub'; %addpath(genpath(path_data_in));

iters=30;
cd(path_data_in)
imagefiles = dir('*.jpg');
cd (base_direc)

for ii=1:num_images
    % Data I/0
    fn_in = imagefiles(ii).name;
    ImIn = double(imread(fn_in));
    num = sscanf(fn_in, '%d.jpg');
    fn_out = strcat('im', num2str(num), '.mat');
    
    x_0=imresize(ImIn,imsize/size(ImIn,1));
    if rgb
        [height, width, depth]=size(x_0);
    else
        [height, width]=size(x_0);
    end
    
    n=length(x_0(:));

    for jj=1:length(num_meas_list)
        path_data_out = strcat('../../reconstructions/',dataset,'/bm3d/meas', ... 
            num2str(num_meas_list(jj)), sprintf('/'));
        
        SamplingRate = samp_rate_list(jj);
        m=round(n*SamplingRate);
        
        M=randn(m,n); % generate random gaussian meas. matrix
        for j = 1:n
            M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
        end
        y=M*x_0(:); % compressively sample image

        %Recover Signal using D-AMP algorithms
        x_hat = DAMP(y,iters,height,width,denoiser1,M);
        size(x_hat);
        
        cd(path_data_out);
        save(fn_out, 'x_hat');
        cd (base_direc);
    end
end
