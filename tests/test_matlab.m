%% Tests Matlab speed at regularized inverse problems
%
% generate canonical test problems:
% http://matrixdepotjl.readthedocs.io/en/latest/regu.html
% julia
% using MatrixDepot
% matrixdepot("deriv2",3,false)
function test_matlab()
addpath('../matlab')
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
 
runtest(A,b,x_true,'deriv2')
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
runtest(A,b,x_true,'shaw')
end

function runtest(A,b,x_true,pn)
%% is it a well-posed, well-conditioned problem?
disp('-------------')
disp([pn,' problem'])
disp(['Condition number: ',num2str(cond(A))])
[U,s] = csvd(A);
figure,picard(U,s,b); title([pn,' Picard plot'])
%% Log mART
x_logmart = logmart(b,A);
assert_gentle(x_logmart,x_true,'logmart')
f = @() logmart(b,A);
disp(['logmart: time to compute [sec.] ',num2str(timeit(f))])
%% Maximum Entropy
%x_python = py.airtools.maxent.maxent(A,b,0.00002)
%py.numpy.testing.assert_array_almost_equal(x_python,x_true)

x_maxent = maxent(A,b,0.001);  
assert_gentle(x_maxent,x_true,'maxent')
f = @() maxent(A,b,0.001);
disp(['maxent: time to compute [sec.] ',num2str(timeit(f))])
%% Kaczmarz ART
%x_python = py.airtools.kaczmarz.kaczmarz(A,b,200)[0]
%py.numpy.testing.assert_array_almost_equal(x_python,x_true)

x_kaczmarz = kaczmarz(A,b,250);
assert_gentle(x_kaczmarz,x_true,'maxent')
f = @() kaczmarz(A,b,250);
disp(['kaczmarz: time to compute [sec.] ',num2str(timeit(f))])
end

function assert_gentle(x,xt,fname)
maxerr = max(abs(x - xt));
if maxerr > max(xt(abs(xt)>0))*0.1 % 10% error
    warning([fname,': too large error from true  '])
    disp(x)
end
end %function