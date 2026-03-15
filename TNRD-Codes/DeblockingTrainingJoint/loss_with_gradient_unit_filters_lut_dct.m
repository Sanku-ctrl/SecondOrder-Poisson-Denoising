function [loss, grad] = loss_with_gradient_unit_filters_lut_dct(vcof)
global G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS T Q_UPPER Q_LOWER DCT_MTX
tic;
clean = G_TRUTH;
input = INPUT;
Q_upper = Q_UPPER;
Q_lower = Q_LOWER;
basis = BASIS;
R = size(clean,2);
r = PATCH_SIZE;
c = PATCH_SIZE;

dct_mtx = DCT_MTX{1};
dct_t   = DCT_MTX{2};
mfs = MFS;
NumW = mfs.NumW;
CropMtx = T;
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
parfor i = 1:filtN
    x_cof = cof_beta(:,i);
    filter = basis*x_cof;
    f_norms(i) = norm(filter);
    filter = filter/norm(filter);
    K{i} = reshape(filter,filter_size,filter_size);
end
%% update mfs
MFsALL = updateMFs(mfs, weights, filtN);
%% do a gradient descent step for all samples
grad_l_x = zeros(size(input));
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
%     dct_x = blockproc(x,[bs bs],dct);
    dct_x = dct_mtx*x(:);
    t_dct_x = max(Q_lower(:,samp), min(Q_upper(:,samp), dct_x(:)));
%     z = blockproc(reshape(t_dct_x,r,c),[bs bs],invdct);
    z = dct_t*t_dct_x;
    
    e = CropMtx*z(:) - clean(:,samp);
    loss = loss + sum(e.^2)/R;
    
    grad_l_z = 2/R*CropMtx'*e;
    % search elements NOT truncated
    ind = (abs(t_dct_x - dct_x(:)) < 1e-12);
%     fprintf('total: %f\n', sum(ind)/r/c*100);
%     t = blockproc(reshape(grad_l_z,r,c),[bs bs],dct).*reshape(ind,r,c);
%     glx = blockproc(t,[bs bs],invdct);
%     grad_l_x(:,samp) = glx(:);
    grad_l_x(:,samp) = dct_t*(ind.*(dct_mtx*grad_l_z));
end
% t1 = toc
%% second way
grad_loss_beta = 0;
grad_loss_weights = 0;
psz = PATCH_SIZE;
pd  = (filter_size-1)/2;
parfor samp = 1:R
    grad_ln_beta = zeros(m,filtN);
    grad_ln_ws   = zeros(NumW,filtN);
    x = reshape(input(:,samp),psz,psz);
    xl = padarray(x, [pd,pd], 'both', 'symmetric');
    v = reshape(grad_l_x(:,samp),psz,psz);
    for i=1:filtN
        k = K{i};
        %% part 1
        kx = imfilter(x,k,'symmetric');
%         [Nkx, N2kx, GW] = MappingAndGrad(kx(:), means, precision, weights(:,i));
        [Nkx,GW,N2kx]   = lut_eval(kx(:)', mfs.offsetD, mfs.step, MFsALL{i}.P, mfs.G, MFsALL{i}.GX, 0);
        Nkx = reshape(Nkx,r,c);
        N2kx = reshape(N2kx,r,c);
        t = convolution_transposeOMP(v,rot90(rot90(k)),psz,psz);
        temp = N2kx.*reshape(t,psz,psz);
        p1 = conv2(xl,rot90(rot90(temp)),'valid');
        %% part 2
        Nkxl = padarray(Nkx, [pd pd], 'both', 'symmetric');
        p2 = conv2(Nkxl,rot90(rot90(v)),'valid');

        gk = p1 + rot90(rot90(p2));
        grad_ln_beta(:,i) = -(eye(m) - cof_beta(:,i)*cof_beta(:,i)'/f_norms(i)^2)/f_norms(i)*basis'*gk(:);
        
        grad_ln_ws(:,i) = -GW*t;
    end
    grad_loss_beta = grad_loss_beta + grad_ln_beta;
    grad_loss_weights = grad_loss_weights + grad_ln_ws;
end
grad = [grad_loss_beta(:);grad_loss_weights(:)];
toc