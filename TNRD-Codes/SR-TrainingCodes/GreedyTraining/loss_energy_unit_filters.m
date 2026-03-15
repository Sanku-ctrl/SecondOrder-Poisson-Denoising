function loss = loss_energy_unit_filters(vcof)
global F_NOISE G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS T SRmtx
noisy = F_NOISE;
clean = G_TRUTH;
input = INPUT;
basis = BASIS;
R = size(noisy,2);
r = PATCH_SIZE;
c = PATCH_SIZE;

mfs = MFS;
NumW = mfs.NumW;
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
x = zeros(size(input));
A = SRmtx;
A_t = A';
parfor samp = 1:R
    u = input(:,samp);
    f = noisy(:,samp);
    g = p*A_t*(A*u - f);
    g = reshape(g,r,c);
    for i=1:filtN
        Ku = imfilter(reshape(u,r,c),K{i},'symmetric');
        Ne1 = lut_eval(Ku(:)', mfs.offsetD, mfs.step, MFsALL{i}.P, 0, 0, 0);
%         Ne1 = Mapping(Ku(:), means, precision, weights(:,i));
        Ne1 = reshape(Ne1,r,c);
        g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x(:,samp) = u - g(:);
end
% t1 = toc
loss = sum(sum((T*x - clean).^2))/R;