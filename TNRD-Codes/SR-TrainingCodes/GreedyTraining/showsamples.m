function showsamples(n)
global U0 G_TRUTH INPUT T
R = size(U0,2);
indperm = randperm(R);
idx = indperm(1:n);

noisy = U0(:,idx);
clean = G_TRUTH(:,idx);
recover = T*INPUT(:,idx);
patch_size = sqrt(size(G_TRUTH,1));
sfigure(1);
displayDictionaryElementsAsImage([clean noisy recover],3,n,patch_size,patch_size,0);
drawnow;