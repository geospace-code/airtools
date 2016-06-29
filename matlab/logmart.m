%solve y=Ax using parallel log-ent mart. 
function [ x,y_est,chi2,i ] = logmart( y,A,relax,x0,sigma,max_iter )
%  Displays delta Chisquare.
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
assert(iscolumn(y),'y must be a column vector')
%% set defaults
if (nargin<6), max_iter=200.; end
if (nargin<5), sigma=ones(size(y)); end
if (nargin<4) || isempty(x0)
    %{
    x=(A'*y)./sum(A(:));
    xA=A*x;
    x=x.*max(y(:))/max(xA(:));
%    max(x(:))
    %}
    x = zeros(size(y));
else
    x=x0;
end
if (nargin<3), relax=1; end
%% make sure there are no 0's in y
y(y<=1e-8)=1e-8;

% W=sigma;
% W=linspace(1,0,size(A,1))';
% W=rand(size(A,1),1);
W=ones(size(A,1),1);
W=W./sum(W);

i=0;
done=false;
arg= ((A*x - y)./sigma).^2;
chi2 = sqrt( sum(arg(:)) );

while ~done
%%  iterate solution, plot estimated data (diag elems of x#A)
  xold=x;
  i=i+1;
  t=min(1./(A*x));
  C=(relax*t*( 1-((A*x)./y) ));
  x=x./(1-x.*(A'*(W.*C)));
%% monitor solution
  chiold=chi2;
  chi2 = sqrt( sum(((A*x - y)./sigma).^2) );
%   dchi2=(chi2-chiold);
  done= ((chi2>chiold) && (i>2)) || (i==max_iter) || (chi2<0.7);

%figure(9); clf; hold off;
%Nest=reshape(x,69,83);
%imagesc(Nest); caxis([0,1e11]);
%set(gca,'YDir','normal'); set(gca,'XDir','normal');
%pause;

end %while
x=xold;
if nargout>1
    y_est=(A*x);
end
end %function

