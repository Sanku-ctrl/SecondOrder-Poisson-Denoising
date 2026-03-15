function showsamples(n, u0, g, x)
R = size(u0,2);
indperm = randperm(R);
idx = indperm(1:n);

noisy = u0(:,idx);
clean = g(:,idx);
recover = x(:,idx);
patch_size = sqrt(size(g,1));
sfigure(1);
displayDictionaryElementsAsImage([clean noisy recover],3,n,patch_size,patch_size,0);
drawnow;