clear all; close all;
filter_size = 5;
m = filter_size^2 - 1;
filtN = 24;
BASIS   = gen_dct2(filter_size);
BASIS   = BASIS(:,2:end);
KernelPara.fsz = filter_size;
KernelPara.filtN = filtN;
KernelPara.basis = BASIS;
load training_5x5_400_180x180_s=3.mat;
%% MFs means and precisions
check_s = 5;
trained_model = save_trained_model(cof, MFS, check_s, KernelPara);
% t = trained_model{2}.mfsAll{11}.P;
% plot(MFS.D, t);grid on;
% rho = RecoverRho(t, MFS.step, length(MFS.D));
% figure; plot(MFS.D, rho)
% grid on;drawnow;
for s = 1:check_s
    mfsAll = trained_model{s}.mfsAll;
    sfigure(100);
    DisplayFilters(trained_model{s}.K,2,12);drawnow;
    for i=1:filtN
        sfigure(i);
        subplot(1,2,1);
        plot(MFS.D, mfsAll{i}.P);grid on;drawnow;
        
        rho = RecoverRho(mfsAll{i}.P, MFS.step, length(MFS.D));
        subplot(1,2,2);
        plot(MFS.D, rho);grid on;drawnow;
%         pause;
    end
    pause;
    close all;
end
return;
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