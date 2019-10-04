module art

use iso_fortran_env, only: wp=>real64

implicit none

contains

pure subroutine logmart(A,b,relax,x0,sigma,max_iter, x)
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

! --- parameter check
real(wp), intent(in) :: A(:,:), b(:)
real(wp), optional, value :: relax, sigma
real(wp), intent(in),optional :: x0(:)
integer, optional, value :: max_iter
real(wp), intent(out) :: x(:)

real(wp), dimension(size(b)) :: W(size(b)), x_prev,c, op_b
integer :: i
real(wp) :: t,chi2,chiold


if (.not.size(A,1) == size(b)) error stop 'A and b row numbers must match'
if (any(A<0)) error stop 'A must be non-negative'
if (any(b<0)) error stop 'b must be non-negative'
op_b = b
! --- make sure there are no 0's in b
where(op_b <= 1e-8) op_b = 1e-8_wp

! --- set defaults
if (.not.present(relax)) relax = 1
if (.not.present(max_iter)) max_iter = 200
if (.not.present(sigma))  sigma = 1

if (.not.present(x0)) then
  x  = matmul(transpose(A), op_b) / sum(A)
  x  = x * maxval(op_b) / maxval(matmul(A, x))
else
  x = x0
endif



! W=sigma;
! W=linspace(1,0,size(A,1))';
! W=rand(size(A,1),1);
W = 1
W = W / sum(W)

! --- iterate solution
chi2 = chi_squared(A, op_b, x, sigma)

do i = 1, max_iter
  x_prev = x
  t = minval(1/matmul(A,x))
  C = relax*t*(1-(matmul(A,x)/op_b))
  x = x / (1-x*matmul(transpose(A),W*C))
! monitor solution
  chiold = chi2
  chi2 = chi_squared(A, op_b, x, sigma)
  if (chi2 > chiold .and. i > 2) exit
enddo

x = x_prev

end subroutine logmart


pure real(wp) function chi_squared(A, b, x, sigma)
real(wp), intent(in) :: A(:,:), b(:), x(:), sigma
chi_squared = sqrt(sum(((matmul(A,x) - b) / sigma)**2))

end function chi_squared

end module art
