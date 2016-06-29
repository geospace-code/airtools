%% Tests Matlab speed at regularized inverse problems
%
% generate test problems:
% julia
% using MatrixDepot
% matrixdepot("deriv2",3,false)

A = [-0.0277778, -0.0277778, -0.00925926;
     -0.0277778, -0.0648148, -0.0277778;
     -0.00925926,-0.0277778, -0.0277778 ];
b = [-0.01514653483985129;
     -0.03474793286789414;
     -0.022274315940957783];
x_true = [0.09622504486493762;
          0.28867513459481287;
          0.48112522432468807];

addpath('../matlab')
%% is it a well-posed problem?
disp(['Condition number: ',num2str(cond(A))])
[U,s,V] = csvd(A);
picard(U,s,b)
%% Log mART
x_logmart = logmart(b,A,1);
assert(all(abs(x_logmart-x_true)<1e-6),'logmart: too large error from true')
%% Maximum Entropy
%x_python = py.airtools.maxent.maxent(A,b,0.00002)
%py.numpy.testing.assert_array_almost_equal(x_python,x_true)

x_maxent = maxent(A,b,0.00002);  
assert(all(abs(x_maxent-x_true)<1e-6),'maxent: too large error from true')
f = @() maxent(A,b,0.00002);
disp(['maxent time to compute [sec.] ',num2str(timeit(f))])
%% Kaczmarz ART
%x_python = py.airtools.kaczmarz.kaczmarz(A,b,200)[0]
%py.numpy.testing.assert_array_almost_equal(x_python,x_true)

x_kaczmarz = kaczmarz(A,b,250);
assert(all(abs(x_kaczmarz-x_true)<1e-6),'kaczmarz: too large error from true')
f = @() kaczmarz(A,b,250);
disp(['kaczmarz time to compute [sec.] ',num2str(timeit(f))])