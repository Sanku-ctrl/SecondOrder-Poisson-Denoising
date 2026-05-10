function [y, gx, gw] = MappingAndGrad(x, means, precision, weights)
x_mu = bsxfun(@minus, x', means);
gw = exp((-0.5*precision)*x_mu.^2);
q = bsxfun(@times, gw, weights);
gx = -precision*sum(bsxfun(@times,q,x_mu),1);
y = sum(q,1);