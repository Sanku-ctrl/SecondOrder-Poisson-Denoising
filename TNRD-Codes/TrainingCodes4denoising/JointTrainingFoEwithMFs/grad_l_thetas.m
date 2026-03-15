function grad = grad_l_thetas(AllNetWorks, trained_model, input, noisy, basis, R, psz, KernelPara, mfs, s)
NumW = mfs.NumW;
filter_size = KernelPara.fsz;
m = filter_size^2 - 1;
filtN = KernelPara.filtN;
pd  = (filter_size-1)/2;
r = psz;
c = psz;

grad_l_u = AllNetWorks{s}.grad_l_u;
model = trained_model{s};
MFsALL = model.MFsALL;
K = model.K;
p = model.p;
cof_beta = model.cof_beta;
f_norms  = model.f_norms;

grad_loss_beta = 0;
grad_loss_weights = 0;
parfor samp = 1:R
    grad_ln_beta = zeros(m,filtN);
    grad_ln_ws   = zeros(NumW,filtN);
    x = reshape(input(:,samp),r,c);
    xl = padarray(x, [pd,pd], 'both', 'symmetric');
    v = reshape(grad_l_u(:,samp),r,c);
    for i=1:filtN
        k = K{i};
        %% part 1
        kx = imfilter(x,k,'symmetric');
        [Nkx,GW,N2kx]   = lut_eval(kx(:)', mfs.offsetD, mfs.step, MFsALL{i}.P, mfs.G, MFsALL{i}.GX, 0);
        Nkx = reshape(Nkx,r,c);
        N2kx = reshape(N2kx,r,c);
        t = convolution_transposeOMP(v,rot90(rot90(k)),r,c);
        temp = N2kx.*reshape(t,r,c);
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
grad_loss_p = -p*sum(sum((input - noisy).*grad_l_u));
grad = [grad_loss_beta(:);grad_loss_p;grad_loss_weights(:)];