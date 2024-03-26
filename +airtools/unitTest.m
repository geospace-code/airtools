% generate canonical test problems:
% http://matrixdepotjl.readthedocs.io/en/latest/regu.html
% julia
% using MatrixDepot
% matrixdepot("deriv2",3,false)

classdef unitTest < matlab.unittest.TestCase

properties (TestParameter)
name = {"deriv2", "shaw"};
end

properties
  TestData
end

methods(TestMethodSetup)
function setup_arrays(tc)

tc.TestData.deriv2.A = [-0.0277778, -0.0277778, -0.00925926;
     -0.0277778, -0.0648148, -0.0277778;
     -0.00925926,-0.0277778, -0.0277778 ];
tc.TestData.deriv2.b = [-0.01514653483985129;
     -0.03474793286789414;
     -0.022274315940957783];
tc.TestData.deriv2.x_true = [0.09622504486493762;
          0.28867513459481287;
          0.48112522432468807];

tc.TestData.shaw.A = [ 0.00289221  0.0536337  0.456086   0.460076  ;
      0.0536337   0.209549   2.68152    0.456086  ;
      0.456086    2.68152    0.209549   0.0536337 ;
      0.460076    0.456086   0.0536337  0.00289221];
tc.TestData.shaw.b = [ 0.875268;
      3.14161 ;
      3.0465  ;
      0.682303];
tc.TestData.shaw.x_true = [  0.398666;
            0.977629;
            0.942325;
            0.851816];

end
end

methods(Test)


function test_condition(tc, name)
%% well-posed, well-conditioned problem?
A = tc.TestData.(name).A;
b = tc.TestData.(name).b;
[U,s] = airtools.csvd(A);
airtools.picard(U, s, b);
disp("Condition #: " + num2str(cond(A)))
end

function test_inverse(tc, name)
A = tc.TestData.(name).A;
b = tc.TestData.(name).b;
x_true = tc.TestData.(name).x_true;

x_inv = A\b;
tc.verifyEqual(x_inv, x_true, 'RelTol', 0.005)
end

function test_pseudoinv(tc, name)
A = tc.TestData.(name).A;
b = tc.TestData.(name).b;
x_true = tc.TestData.(name).x_true;

x_pinv = pinv(A)*b;
tc.verifyEqual(x_pinv, x_true, RelTol=0.005)
end

function test_logmart(tc, name)
A = tc.TestData.(name).A;
b = tc.TestData.(name).b;
x_true = tc.TestData.(name).x_true;

tc.assumeGreaterThanOrEqual(b, 0)

x_logmart = airtools.logmart(b,A);
tc.verifyEqual(x_logmart, x_true, RelTol=0.1)
end

function test_maxent(tc, name)
A = tc.TestData.(name).A;
b = tc.TestData.(name).b;
x_true = tc.TestData.(name).x_true;

x_maxent = airtools.maxent(A,b,0.001);
tc.verifyEqual(x_maxent, x_true, RelTol=0.05)
end

function test_maxent_python(tc, name)
A = tc.TestData.(name).A;
b = tc.TestData.(name).b;
x_true = tc.TestData.(name).x_true;

x_python = py.airtools.maxent.maxent(A,b,0.00002)
tc.verifyEqual(x_python, x_true)
end

function test_kart(tc,name)
A = tc.TestData.(name).A;
b = tc.TestData.(name).b;
x_true = tc.TestData.(name).x_true;
% x_python = py.airtools.kaczmarz.kaczmarz(A,b,200)[0]
% py.numpy.testing.assert_array_almost_equal(x_python,x_true)
%
x_kaczmarz = airtools.kaczmarz(A,b,250);
tc.verifyEqual(x_kaczmarz, x_true, RelTol=0.05)
end

end

end
