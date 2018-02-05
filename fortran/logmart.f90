! solve b=Ax using parallel log-ent mart.
! Matlab by Joshua Semeter
! Fortran by Michael Hirsch

program test_logmart
use iso_fortran_env, only: wp=>real64
implicit none



integer, parameter :: N=3
real(wp) :: A(N,N)
real(wp), parameter :: x_true(N)=[5,5,5] 
real(wp),parameter :: pi = 4.*atan(1.), errtol=0.05_wp
real(wp), dimension(N) :: x, noise, b,bias

call init_random_seed()

A = reshape([1,0,0,0,1,0,0,0,1],[N,N])


! ---- noisy observation
call randn(bias)
print*,'bias',bias
A = A! * spread(bias,2,N)

call randn(noise)
print*,'noise',noise
b = matmul(A,x_true) + 0.1_wp*noise

! ---- inversion
call logmart(A,b,x=x) 

! --- check estimate
if (all(abs(x-x_true) < errtol*maxval(x_true))) then
  print*,'logmart SUCCESS'
else
  print*,x
  print*,'larger than',errtol*100,' % error'
  stop 
endif            

contains

pure subroutine logmart(A,b,relax,x0,sigma,max_iter,x)
! delta Chisquare.
! stopped if Chisquare increases.
!
! Inputs
! ------
! A: NxM array
! b: N vector
! relax: user specified relaxation constant	(default is 20.)
! x0: user specified initial guess (Nx1 vector)  (default is backproject y, i.e., y#A)
! max_iter: user specified max number of iterations (default is 20)
!
!
! Outputs
! -------
! x: N vector
!
!
!    AUTHOR:	Joshua Semeter
!    LAST MODIFIED:	5-2015
!
!      Simple test problem
!    A = diag([5, 5, 5])
!    x = [1,2,3]
!    b = A*x

use iso_fortran_env, only: wp=>real64


! --- parameter check
real(wp), intent(in) :: A(:,:), b(:)
real(wp), optional, value :: relax
real(wp), intent(in),optional :: x0(:), sigma(:)
integer, optional, value :: max_iter
real(wp), intent(out) :: x(:)

real(wp), dimension(size(b)) :: xA, op_sigma, W, op_b, arg,xold,c
integer :: i
logical :: done
real(wp) :: t,chi2,chiold

op_b = b

if (.not.size(A,1) == size(b)) error stop 'A and b row numbers must match'

! --- set defaults
if (.not.present(relax)) relax = 1._wp
if (.not.present(max_iter)) max_iter = 200

if (.not.present(x0)) then
  x  = matmul(transpose(A), b) / sum(A)
  xA = matmul(A, x)
  x  = x * maxval(b) / maxval(xA)
else
  x = x0
endif

if (.not.present(sigma)) then
  op_sigma = 1._wp
else
  op_sigma = sigma
endif

! --- make sure there are no 0's in b
where(op_b<=1e-8) op_b = 1e-8_wp

! W=sigma;
! W=linspace(1,0,size(A,1))';
! W=rand(size(A,1),1);
W = 1
W = W / sum(W)

! --- iterate solution
i=0
done=.false.
arg= ((matmul(A,x) - op_b) / op_sigma)**2
chi2 = sqrt(sum(arg))

do while (.not.done)
  i = i+1
  xold = x
  xA = matmul(A,x)
  t = minval(1/xA)
  C = relax*t*(1-(xA/b))
  x = x / (1-x*matmul(transpose(A),W*C))
! monitor solution
  chiold = chi2
  chi2 = sqrt( sum(((xA - b)/op_sigma)**2) )
  ! dchi2=(chi2-chiold)
  done = ((chi2>chiold) .and. (i>2)) .or. (i==max_iter) .or. (chi2<0.7)
enddo

x = xold

end subroutine logmart


subroutine randn(noise)
! implements Box-Muller Transform
! https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
!
! Output:
! noise: Gaussian noise vector

real(wp),intent(out) :: noise(:)
real(wp),dimension(size(noise)) :: u1, u2 

call random_number(u1)
call random_number(u2)

noise = sqrt(-2._wp * log(u1)) * cos(2._wp * pi * u2)

end subroutine randn


subroutine init_random_seed()
! don't call this function repeatedly in your program.
! The time resolution of int32 clock isn't so high, and the seed only
! accepts int32, despite nice clock performance with int64
integer :: n,i, clock
integer, allocatable :: seed(:)


call random_seed(size=n)
allocate(seed(n))
call system_clock(count=clock)
seed = clock + 37 * [ (i - 1, i = 1, n) ]
call random_seed(put=seed)

end subroutine

end program
