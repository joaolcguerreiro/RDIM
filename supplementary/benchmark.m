close all; clear all;

% Directories containing images
generated_dir = 'results_TIFF';   % folder with generated TIFFs
gt_dir = 'GT';               % folder with ground truth PNGs

% List of files (assume same filenames in both folders)
generated_files = dir(fullfile(generated_dir, '*.tiff'));
gt_files = dir(fullfile(gt_dir, '*.png'));

% Sort to make sure matching order
generated_files = sort({generated_files.name});
gt_files = sort({gt_files.name});

total_psnr = 0;
total_ssim = 0;
num_images = numel(generated_files);

for i = 1:num_images
    % Read images
    generated_img = im2single(imread(fullfile(generated_dir, generated_files{i}))) / 255;
    gt_img = im2single(imread(fullfile(gt_dir, gt_files{i})));

    % Compute PSNR and SSIM
    psnr_val = psnr(generated_img, gt_img);
    ssim_val = ssim(generated_img, gt_img);

    % Accumulate
    total_psnr = total_psnr + psnr_val;
    total_ssim = total_ssim + ssim_val;
end

% Average
qm_psnr = total_psnr / num_images;
qm_ssim = total_ssim / num_images;

fprintf('PSNR: %f SSIM: %f\n', qm_psnr, qm_ssim);
