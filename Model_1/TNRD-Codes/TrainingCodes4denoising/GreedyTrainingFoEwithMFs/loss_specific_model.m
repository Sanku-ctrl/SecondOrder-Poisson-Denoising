function [loss, x_star] = loss_specific_model(noisy, clean, input, KernelPara, mfs, T, boolshow)
global PATCH_SIZE
R = size(noisy,2);
r = PATCH_SIZE;
c = PATCH_SIZE;

means = mfs.means;
precision = mfs.precision;
NumW = mfs.NumW;

basis = KernelPara.basis;
vcof = KernelPara.cof;
% tic;
filter_size = KernelPara.fsz;
m = filter_size^2 - 1;
filtN = KernelPara.filtN;
part1 = vcof(1:filtN*m);
cof_beta = reshape(part1,m,filtN);
% weight theta
% part2 = vcof(filtN*m+1:filtN*(m+1));
% theta = exp(part2);
part3 = vcof(filtN*m+1);
p = exp(part3);
part4 = vcof(filtN*m+2:end);
weights = reshape(part4,NumW,filtN);

% construct filters
% filters = basis*cof_beta;
% sfigure(1);
% DisplayFilters(filters',2,12,filter_size,theta);drawnow;

%% unit norm filters
K = cell(filtN,1);
f_norms = zeros(filtN,1);
filters = [];
parfor i = 1:filtN
    x_cof = cof_beta(:,i);
    filter = basis*x_cof;
    f_norms(i) = norm(filter);
    filter = filter/norm(filter);
    K{i} = reshape(filter,filter_size,filter_size);
    
    filters = [filters, filter];
end
%% update mfs
MFsALL = updateMFs(mfs, weights, filtN);
%% do a gradient descent step for all samples
x_star = zeros(size(input));
parfor samp = 1:R
    u = input(:,samp);
    f = noisy(:,samp);
    g = (u - f)*p;
    g = reshape(g,r,c);
    for i=1:filtN
        Ku = imfilter(reshape(u,r,c),K{i},'symmetric');
        Ne1 = lut_eval(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P, 0, 0, 0);
        Ne1 = reshape(Ne1,r,c);
        g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x_star(:,samp) = u - g(:);
end
% t1 = toc
loss = sum(sum((T*x_star - clean).^2))/R;
cmap = hsv(filtN);
if boolshow
    close all;    
    x = mfs.D;
    x_mu = bsxfun(@minus, x, means);
    t = bsxfun(@times, x_mu.^2, -0.5*precision);
    gw = exp(t);
    sfigure(100);
    hold on;
    for i=1:filtN
        w = weights(:,i);
        q = bsxfun(@times, gw, w);
        p = sum(q,1);
        plot(x,p,'Color',cmap(i,:));drawnow;
    end
    grid on;
%     plot(x,2*x./(1+x.^2),'k.')
    hold off;
    sfigure(101);
    DisplayFilters(filters',2,12,filter_size);
    drawnow;
end