%% Tests Matlab speed at regularized inverse problems
%
% generate canonical test problems:
% http://matrixdepotjl.readthedocs.io/en/latest/regu.html
% julia
% using MatrixDepot
% matrixdepot("deriv2",3,false)
function test_matlab()
addpath('../matlab')
methods = {'inv','pinv','logmart','maxent','kaczmarz'};
%% deriv2     
A = [-0.0277778, -0.0277778, -0.00925926;
     -0.0277778, -0.0648148, -0.0277778;
     -0.00925926,-0.0277778, -0.0277778 ];
b = [-0.01514653483985129;
     -0.03474793286789414;
     -0.022274315940957783];
x_true = [0.09622504486493762;
          0.28867513459481287;
          0.48112522432468807];
 
t_deriv2 = runtest(A,b,x_true,'deriv2');
plott(t_deriv2,methods)
%% shaw
A = [ 0.00289221  0.0536337  0.456086   0.460076  ;
      0.0536337   0.209549   2.68152    0.456086  ;
      0.456086    2.68152    0.209549   0.0536337 ;
      0.460076    0.456086   0.0536337  0.00289221];
b = [ 0.875268;
      3.14161 ;
      3.0465  ;
      0.682303];
x_true = [  0.398666;
            0.977629;
            0.942325;
            0.851816];
t_shaw = runtest(A,b,x_true,'shaw');
plott(t_shaw,methods)
end

function t = runtest(A,b,x_true,pn)
%% is it a well-posed, well-conditioned problem?
[U,s] = csvd(A);
figure,subplot(1,2,1)
picard(U,s,b); title([pn,' Picard plot  Cond. #: ',num2str(cond(A))])
%% inverse
x_inv = A\b;
assert_gentle(x_inv,x_true,'inv')
f = @() A\b;
t(1) = timeit(f);
%% pseudoinv
x_pinv = pinv(A)*b;
assert_gentle(x_pinv,x_true,'inv')
f = @() pinv(A)*b;
t(2) = timeit(f);
%% Log mART
x_logmart = logmart(b,A);
assert_gentle(x_logmart,x_true,'logmart')
f = @() logmart(b,A);
t(3) = timeit(f);
%% Maximum Entropy
%x_python = py.airtools.maxent.maxent(A,b,0.00002)
%py.numpy.testing.assert_array_almost_equal(x_python,x_true)

x_maxent = maxent(A,b,0.001);  
assert_gentle(x_maxent,x_true,'maxent')
f = @() maxent(A,b,0.001);
t(4) = timeit(f);
%% Kaczmarz ART
%x_python = py.airtools.kaczmarz.kaczmarz(A,b,200)[0]
%py.numpy.testing.assert_array_almost_equal(x_python,x_true)

x_kaczmarz = kaczmarz(A,b,250);
assert_gentle(x_kaczmarz,x_true,'maxent')
f = @() kaczmarz(A,b,250);
t(5) = timeit(f);
end

function plott(t,meth)
   ax = subplot(1,2,2);
   stem(ax,t)
   xlim(ax,[0.5,length(meth)+0.5])
   set(ax,'xtick',1:length(meth))
   set(ax,'xticklabel',meth)
   ylabel(ax,'time to compute [sec]')
   title(ax,'Computation Time')
end

function assert_gentle(x,xt,fname)
err = abs(x - xt);
if any(err > max(xt(abs(xt)>0))*0.05) % 5% error
    warning([fname,': too large error from true'])
    disp(x)
end
end %function