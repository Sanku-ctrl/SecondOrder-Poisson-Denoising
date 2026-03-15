function showsamples(n)
global F_NOISE G_TRUTH INPUT T
R = size(F_NOISE,2);
indperm = randperm(R);
idx = indperm(1:n);

offset = 128;
noisy = T*F_NOISE(:,idx)+offset;
clean = G_TRUTH(:,idx)+offset;
recover = T*INPUT(:,idx)+offset;
patch_size = sqrt(size(G_TRUTH,1));
sfigure(1);
displayDictionaryElementsAsImage([clean noisy recover],3,n,patch_size,patch_size,0);
drawnow;