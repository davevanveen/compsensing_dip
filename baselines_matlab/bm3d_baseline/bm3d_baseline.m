% Demonstrates compressively sampling and D-AMP recovery of an image.

% NOTE: Need to fix absolute paths, add user-specified arguments
%      i.e. code not currently user-friendly 

clear all

bm3d_path = '/home/dave/Documents/comp_sensing/cs_dip/csdip_code/cs_dip/D-AMP_Toolbox-master/Demos/BM3D';
utils_path = '/home/dave/Documents/comp_sensing/cs_dip/csdip_code/cs_dip/D-AMP_Toolbox-master/Demos/Utils';
data_path = '/home/dave/Documents/comp_sensing/cs_dip/csdip_code/cs_dip/icml_push/data/xray/sub';
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
num_meas_list = [500, 1000, 1500, 2000, 4000, 8000]; %xray
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
        path_data_out = strcat('~/Documents/comp_sensing/cs_dip/csdip_code/cs_dip', ...
        '/icml_push/reconstructions/xray/bm3d/meas', num2str(num_meas_list(jj)), sprintf('/'));
        cd(path_data_out);
    
        SamplingRate = samp_rate_list(jj);
        m=round(n*SamplingRate)
        
        M=randn(m,n); % generate random gaussian meas. matrix
        for j = 1:n
            M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
        end
        y=M*x_0(:); % compressively sample image

        %Recover Signal using D-AMP algorithms
        x_hat = DAMP(y,iters,height,width,denoiser1,M);
        
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
