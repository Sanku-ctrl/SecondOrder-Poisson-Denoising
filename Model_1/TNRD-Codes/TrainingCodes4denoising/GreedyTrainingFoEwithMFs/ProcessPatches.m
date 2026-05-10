function samples = ProcessPatches(noisy,R,psz,bsz)
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
crop = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
samples = [];
parfor i=1:R
    y = reshape(noisy(:,i),psz,psz);
    y = crop(y);
    y = pad(y);
    samples = [samples y(:)];
end