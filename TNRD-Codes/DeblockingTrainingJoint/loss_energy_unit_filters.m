function loss = loss_energy_unit_filters(vcof)
global G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS DIFF_TERM PT Q_UPPER Q_LOWER
clean = G_TRUTH;
input = INPUT;
diffterm = DIFF_TERM;
Q_upper = Q_UPPER;
Q_lower = Q_LOWER;
basis = BASIS;
R = size(clean,2);
r = PATCH_SIZE;
c = PATCH_SIZE;

means = MFS.means;
precision = MFS.precision;
NumW = MFS.NumW;
CropMtx = PT;
% tic;
filter_size = KernelPara.fsz;
m = filter_size^2 - 1;
filtN = KernelPara.filtN;
part1 = vcof(1:filtN*m);
cof_beta = reshape(part1,m,filtN);
% weight theta
% part2 = vcof(filtN*m+1:filtN*(m+1));
% theta = exp(part2);
part4 = vcof(filtN*m+1:end-1);
weights = reshape(part4,NumW,filtN);
part5 = vcof(end);
beta = exp(part5);

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
%% do a gradient descent step for all samples
bs = 8;
T = dctmtx(bs);
invdct = @(block_struct) T' * block_struct.data * T;
dct = @(block_struct) T * block_struct.data * T';
loss = 0;
parfor samp = 1:R
    u = reshape(input(:,samp),r,c);
    g = 0;
    for i=1:filtN
        Ku = imfilter(u,K{i},'symmetric');
        Ne1 = Mapping(Ku(:), means, precision, weights(:,i));
        Ne1 = reshape(Ne1,r,c);
        g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x = u - g + beta*reshape(diffterm(:,samp),r,c);
    
    %% projection to QCS, truncate cofs
    dct_x = blockproc(x,[bs bs],dct);
    t_dct_x = max(Q_lower(:,samp), min(Q_upper(:,samp), dct_x(:)));
    z = blockproc(reshape(t_dct_x,r,c),[bs bs],invdct);
    
    loss = loss + sum((CropMtx*z(:) - clean(:,samp)).^2)/R;
end