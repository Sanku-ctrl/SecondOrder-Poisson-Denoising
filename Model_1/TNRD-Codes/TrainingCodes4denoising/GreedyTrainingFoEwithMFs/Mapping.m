function y = Mapping(x, means, precision, weights)
x_mu = bsxfun(@minus, x', means);
% t = bsxfun(@times, x_mu.^2, -0.5*precision);
% gw = exp(t);
gw = exp((-0.5*precision)*x_mu.^2);
q = bsxfun(@times, gw, weights);
y = sum(q,1)';