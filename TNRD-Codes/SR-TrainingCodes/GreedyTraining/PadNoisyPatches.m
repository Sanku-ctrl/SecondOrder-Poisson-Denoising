function [T, Pinit, Pnoisy] = PadNoisyPatches(init, noisy, R, psz, bsz, scale)
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
padLR = @(x) padarray(x,bndry/scale,'symmetric','both');
Pinit = [];
Pnoisy = [];
parfor i=1:R
    y = reshape(init(:,i),psz,psz);
    y = pad(y);
    Pinit = [Pinit y(:)];
    
    y = reshape(noisy(:,i),psz/scale,psz/scale);
    y = padLR(y);
    Pnoisy = [Pnoisy y(:)];
end

imdims = [psz+2*bsz, psz+2*bsz];
t = bsz;
npixels = prod(imdims);
% truncation/cropping matrix T
[r,c] = ndgrid(1+t:imdims(1)-t, 1+t:imdims(2)-t);
ind_int = sub2ind(imdims, r(:), c(:));
d = zeros(imdims); d(ind_int) = 1;
T = spdiags(d(:),0,npixels,npixels);
T = T(ind_int,:);

% padding matrix P that mirror boundary pixels, symmetric boundary
% condition
% idximg = reshape(1:prod(imdims-2*t),imdims-2*t);
% pad_idximg = padarray(idximg,[t t],'symmetric','both');
% P = sparse((1:npixels)',pad_idximg(:),ones(npixels,1),npixels,prod(imdims-2*t));
% 
% % first truncation, then padding
% PT = P*T;