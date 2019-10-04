module art

implicit none

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

end module art
