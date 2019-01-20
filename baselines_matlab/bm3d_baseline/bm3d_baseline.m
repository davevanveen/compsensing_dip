% Demonstrates compressively sampling and D-AMP recovery of an image.

% NOTE: Need to fix absolute paths, add user-specified arguments
%      i.e. code not currently user-friendly 

clear all

home_path = '~/work/compsensing_dip/baselines_matlab/bm3d_baseline/';
bm3d_path = '~/work/compsensing_dip/baselines_matlab/bm3d_baseline/BM3D/';
utils_path = '~/work/compsensing_dip/baselines_matlab/bm3d_baseline/Utils/';
data_path = '~/work/compsensing_dip/data/xray/sub/';
matrices_path = '~/work/compsensing_dip/measurement_matrices/';
addpath(genpath(bm3d_path));
addpath(genpath(utils_path));
addpath(genpath(data_path));

%Parameters
denoiser1='BM3D';%Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, and BM3D-SAPCA 
iters=30;

cd(data_path)
imagefiles = dir('*.jpg');
num_images = 60;

imsize=256;
num_pixels = imsize^2;
%num_meas_list = [10, 15, 25, 35, 50, 75, 100, 200]; %mnist
num_meas_list = [5]; %xray
samp_rate_list = num_meas_list ./ num_pixels;

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

    
        SamplingRate = samp_rate_list(jj);
        m=round(n*SamplingRate)
        
        %M=randn(m,n); % generate random gaussian meas. matrix
        %for j = 1:n
        %    M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
        %end
	matrix_path_in = strcat(matrices_path, 'fourier_',num2str(num_meas_list(jj)),'.mat');
	M = load(matrix_path_in);
	M = M.M;
	y=M*x_0(:); % compressively sample image
	
        %Recover Signal using D-AMP algorithms
	cd(home_path)        
	x_hat = DAMP(y,iters,height,width,denoiser1,M);
        
	path_data_out = strcat('~/work/compsensing_dip/reconstructions/xray/bm3d/meas', num2str(num_meas_list(jj)), sprintf('/'));
        cd(path_data_out);
        
        save(fn_out, 'x_hat');
    end
end


%D-AMP Recovery Performance
performance1=PSNR(x_0,x_hat);
[num2str(SamplingRate*100),'% Sampling ', denoiser1, '-AMP Reconstruction PSNR=',num2str(performance1)]

%Plot Recovered Signals
subplot(1,2,1);
imshow(uint8(x_0));title('Original Image');
subplot(1,2,2);
imshow(uint8(x_hat));title([denoiser1, '-AMP']);
