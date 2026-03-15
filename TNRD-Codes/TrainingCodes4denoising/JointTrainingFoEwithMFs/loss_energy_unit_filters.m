function loss = loss_energy_unit_filters(vcof)
global F_NOISE G_TRUTH BASIS INPUT PATCH_SIZE KernelPara MFS DIFF_TERM T
noisy = F_NOISE;
clean = G_TRUTH;
input = INPUT;
diffterm = DIFF_TERM;
basis = BASIS;
R = size(noisy,2);
r = PATCH_SIZE;
c = PATCH_SIZE;

means = MFS.means;
precision = MFS.precision;
NumW = MFS.NumW;

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
part4 = vcof(filtN*m+2:end-1);
weights = reshape(part4,NumW,filtN);
part5 = vcof(end);
beta = exp(part5);

% construct filters
% filters = basis*cof_beta;
% sfigure(1);
% DisplayFilters(filters',2,12,filter_size,theta);drawnow;

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
x = zeros(size(input));
parfor samp = 1:R
    u = input(:,samp);
    f = noisy(:,samp);
    g = (u - f)*p;
    g = reshape(g,r,c);
    for i=1:filtN
        Ku = imfilter(reshape(u,r,c),K{i},'symmetric');
        Ne1 = Mapping(Ku(:), means, precision, weights(:,i));
        Ne1 = reshape(Ne1,r,c);
        g = g + imfilter(Ne1,rot90(rot90(K{i})),'symmetric');
    end
    x(:,samp) = u - g(:) + beta*diffterm(:,samp);
end
% t1 = toc
loss = sum(sum((T*x - clean).^2))/R;