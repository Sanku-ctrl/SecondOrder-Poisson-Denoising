function [loss, x_star] = deblocking_dataSets(clean, input, diffterm, Q_upper, Q_lower, model, CropMtx)
[N,R] = size(input);
psz = sqrt(N);
r = psz;
c = psz;

mfsAll = model.mfsAll;
K = model.K;
beta = model.beta;
mfs = model.mfs;
filtN = length(K);

%% do a gradient descent step for all samples
bs = 8;
T = dctmtx(bs);
invdct = @(block_struct) T' * block_struct.data * T;
dct = @(block_struct) T * block_struct.data * T';
loss = 0;
x_star = zeros(size(input));
parfor samp = 1:R
    u = reshape(input(:,samp),r,c);
    g = 0;
    for i=1:filtN
        Ku = imfilter(u,K{i},'symmetric');
        Ne1 = lut_eval(Ku(:)', mfs.offsetD, mfs.step, mfsAll{i}.P, 0, 0, 0);
        Ne1 = reshape(Ne1,r,c);
        g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x = u - g + beta*reshape(diffterm(:,samp),r,c);
    
    %% projection to QCS, truncate cofs
    dct_x = blockproc(x,[bs bs],dct);
    t_dct_x = max(Q_lower(:,samp), min(Q_upper(:,samp), dct_x(:)));
    z = blockproc(reshape(t_dct_x,r,c),[bs bs],invdct);
    
    loss = loss + sum((CropMtx*z(:) - clean(:,samp)).^2)/R;
    
    x_star(:,samp) = z(:);
end