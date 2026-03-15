% clear all; clc;
% load g_new.mat;
function [Q_lower, Q_upper, clean, noisy] = generateDeblockingSamples(R, path, q, sign_ud, sign_lr, sign_lrud)
offset = 128;
bsz = 8;
bs = 8;
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
T = dctmtx(bs);
invdct = @(block_struct) T' * block_struct.data * T;
dct = @(block_struct) T * block_struct.data * T';

% q = 10;
clean = [];
noisy = [];
Q_upper = [];
Q_lower = [];
parfor i=1:R
    file = strcat(path,sprintf('test_%03d.png', i));
    img = double(imread(file));
    jpgfile = sprintf('./temp/test_%03d.jpg', i);
    imwrite(img/255,jpgfile,'jpg','Quality',q);
    JPEG_header_info = jpeg_read(jpgfile);
    Q = JPEG_header_info.quant_tables{1};
    cofQ = JPEG_header_info.coef_arrays{1};
    
    f = double(imread(jpgfile)) - offset;
    % block function
%     multiply = @(block_struct) block_struct.data.*Q;
    s = 0.8;
    minus = @(block_struct) block_struct.data.*Q - 0.5*Q*s;
    plus = @(block_struct) block_struct.data.*Q + 0.5*Q*s;    
    
%     cof = blockproc(cofQ,[bs bs],multiply);
%     recover = blockproc(cof,[bs bs],invdct);
    padQ = SymmetricPadQcof(cofQ, sign_ud, sign_lr, sign_lrud, bs);
    
    lcof = blockproc(padQ,[bs bs],minus);
    ucof = blockproc(padQ,[bs bs],plus);
%     cof = blockproc(pad(img) - offset,[8 8],dct);
%     cof = blockproc(padQ,[bs bs],multiply);
%     recover = blockproc(cof,[bs bs],invdct);
%     center = blockproc(crop(cof),[bs bs],invdct);
    
    clean = [clean img(:)-offset];
%     input = [input recover(:)];
    noisy = [noisy f(:)];
    Q_upper = [Q_upper ucof(:)];
    Q_lower = [Q_lower lcof(:)];
    
%     delta = ucof - cof;
%     fprintf('Not good: %d\n',sum(delta(:) < 0));    
       
%     delta = recover2 - pad(recover);
%     fprintf('diff: [%g, %g]\n',min(delta(:)),max(delta(:)));
end
% [T,noisy] = PadNoisyPatches(noisy,R,psz,bsz);