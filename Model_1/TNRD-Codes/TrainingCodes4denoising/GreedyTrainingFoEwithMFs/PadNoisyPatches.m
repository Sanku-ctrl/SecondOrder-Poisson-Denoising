function [T,samples] = PadNoisyPatches(noisy,R,psz,bsz)
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
samples = [];
parfor i=1:R
    y = reshape(noisy(:,i),psz,psz);
    y = pad(y);
    samples = [samples y(:)];
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


% padding matrix P that replicates boundary pixels
% (also called (zero-flux) Neumann boundary condition)
% idximg = reshape(1:prod(imdims-2*t),imdims-2*t);
% pad_idximg = padarray(idximg,[t t],'replicate','both');
% P = sparse((1:s.npixels)',pad_idximg(:),ones(s.npixels,1),s.npixels,prod(imdims-2*t));



