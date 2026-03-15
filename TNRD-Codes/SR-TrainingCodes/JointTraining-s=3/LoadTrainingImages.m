function [clean, input, noisy] = LoadTrainingImages(R, path, scale)
clean = [];
noisy = [];
input = [];
parfor pic_idx=1:R
    file = strcat(path,sprintf('test_%03d.png', pic_idx));
    img = double(imread(file));
    f = imresize(img, 1/scale, 'bicubic');
    sr = imresize(f, scale, 'bicubic');
    
    clean = [clean img(:)];
    noisy = [noisy f(:)];
    input = [input sr(:)];
end
% randn('seed', 0);
% noisy = clean + scale*randn(size(clean));
% noisy = double(uint8(noisy));