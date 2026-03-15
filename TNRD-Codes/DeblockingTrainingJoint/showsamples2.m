function showsamples2(n, R, F_NOISE, G_TRUTH, x_s, T)
indperm = randperm(R);
idx = indperm(1:n);
offset = 128;
noisy = T*F_NOISE(:,idx)+offset;
clean = G_TRUTH(:,idx)+offset;
recover = T*x_s(:,idx)+offset;
patch_size = sqrt(size(G_TRUTH,1));
sfigure(1);
displayDictionaryElementsAsImage([clean noisy recover],3,n,patch_size,patch_size,0);
drawnow;