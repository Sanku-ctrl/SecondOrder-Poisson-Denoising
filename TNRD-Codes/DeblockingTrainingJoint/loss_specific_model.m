function [loss, x_star] = loss_specific_model(clean, input, Q_upper, Q_lower, KernelPara, mfs, CropMtx, DCT_MTX, boolshow)
global PATCH_SIZE
R = size(input,2);
r = PATCH_SIZE;
c = PATCH_SIZE;
dct_mtx = DCT_MTX{1};
dct_t   = DCT_MTX{2};

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
part2 = vcof(filtN*m+1:end);
weights = reshape(part2,NumW,filtN);
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
loss = 0;
parfor samp = 1:R
    u = reshape(input(:,samp),r,c);
    g = 0;
    for i=1:filtN
        Ku = imfilter(u,K{i},'symmetric');
%         Ne1 = Mapping(Ku(:), means, precision, weights(:,i));
        Ne1 = lut_eval_one_variable(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P);
        Ne1 = reshape(Ne1,r,c);
        g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x = u - g;    
    %% projection to QCS, truncate cofs
    dct_x = dct_mtx*x(:);
    t_dct_x = max(Q_lower(:,samp), min(Q_upper(:,samp), dct_x(:)));
    z = dct_t*t_dct_x;    
    loss = loss + sum((CropMtx*z(:) - clean(:,samp)).^2)/R;
    x_star(:,samp) = z;
end
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