%solve y=Ax using parallel log-entropy MART  (De Pierro 1991)
function [ x,y_est,chi2,i ] = logmart( y,A,relax,x0,sigma,max_iter )
% Program is stopped if Chisquare increases.
% A is NxM array
% Y is Nx1 vector
% returns Mx1 vector
%
% KEYWORDS:
% relax		user specified relaxation constant	(default is 20.)
% x0	user specified initial guess (N vector)
%           (default is backproject y, i.e., y#A)
% max_iter	user specified max number of iterations (default is 20)
%
%AUTHOR:	Joshua Semeter
%LAST MODIFIED:	5-2015

%  Simple test problem
% A=diag([5 5 5]);
% x=[1;2;3];
% y=A*x;
arguments
y (:,1) {mustBeNumeric,mustBeNonnegative}
A (:,:) {mustBeNumeric,mustBeNonnegative}
relax (1,1) {mustBePositive} = 20
x0 {mustBeNumeric,mustBeNonnegative} = []
sigma (1,1) {mustBeNumeric,mustBePositive} = 1
max_iter (1,1) {mustBeInteger, mustBePositive} = 20
end
%% make sure there are no 0's in y
y(y<=1e-8) = 1e-8;

if isempty(x0)
    x=(A'*y)./sum(A(:));
    xA=A*x;
    x=x.*max(y(:))/max(xA(:));
else
    x=x0;
end

% W=sigma;
% W=linspace(1,0,size(A,1))';
% W=rand(size(A,1),1);
W=ones(size(A,1),1);
W=W./sum(W);

chi2 = chi_squared(y, A, x, sigma);

for i = 1:max_iter
%%  iterate solution, plot estimated data (diag elems of x#A)
  xold=x;
  t=min(1./(A*x));
  C=(relax*t*( 1-((A*x)./y) ));
  x=x./(1-x.*(A'*(W.*C)));
%% monitor solution
  chiold=chi2;
  chi2 = chi_squared(y, A, x, sigma);
%   dchi2=(chi2-chiold);
  if ((chi2>chiold) && (i>2)), break, end  %  || (chi2<0.7)

%figure(9); clf; hold off;
%Nest=reshape(x,69,83);
%imagesc(Nest); caxis([0,1e11]);
%set(gca,'YDir','normal'); set(gca,'XDir','normal');
%pause;

end % for
x=xold;
if nargout>1
    y_est=(A*x);
end
end %function

function chi2 = chi_squared(y, A, x, sigma)
chi2 = sqrt( sum(((A*x - y)./sigma).^2) );
end
