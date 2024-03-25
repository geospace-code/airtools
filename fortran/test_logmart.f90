! solve b=Ax using parallel log-ent mart.
! Matlab by Joshua Semeter
! Fortran by Michael Hirsch

program test_logmart

use, intrinsic:: iso_fortran_env, only: stderr=>error_unit
use art, only: logmart
use random_utils, only: randn, wp

implicit none

integer, parameter :: N=4, M=4
real(wp), dimension(N,M) :: A1, A2
real(wp) :: x(M)
logical :: ok=.true., add_noise = .true.

call random_init(.false., .false.)

A1 = reshape([5,0,0,0, &
              0,5,0,0, &
              0,0,5,0, &
              0,0,0,5], shape(A1), order=[2,1])

A2 = reshape([0,1,2,3, &
              1,0,1,2, &
              2,1,0,1, &
              3,2,1,0], shape(A2), order=[2,1])

x = [1._wp, 3._wp, 0.5_wp, 2._wp]

if (.not. run_test(A1, x, 20,add_noise)) then
  ok = .false.
  write(stderr, '(a)') 'failed on identity test'
endif

if (.not. run_test(A2, x, 2000, add_noise)) then
  ok = .false.
  write(stderr, '(a)') 'failed on Fiedler test'
endif

if (.not. ok) error stop

print *, 'OK: logmart'

contains

logical function run_test(A, x, max_iter, add_noise)

real(wp), intent(in) :: A(:,:), x(:)
logical, intent(in) :: add_noise
integer, intent(in) :: max_iter

real(wp), parameter :: errtol=0.05_wp
real(wp), dimension(size(A,1)) :: noise, x_est, b

run_test = .true.

block
integer :: i
print '(a)', 'A ='
do i = 1,size(A,1)
  print '(3F10.3)', A(i,:)
enddo
end block

! ---- noisy observation
b = matmul(A,x)

if (add_noise) then
  call randn(noise)
  noise = 0.01_wp * noise
  print '(/,A,3F10.3)', 'noise',noise
  b = b + noise
endif

! ---- inversion
call logmart(A,b, max_iter=max_iter, x=x_est)

! --- check estimate
if (any(abs(x_est-x) > errtol*maxval(x))) then
  print *,x_est
  write (stderr,*) 'larger than',errtol*100,' % error'
  run_test = .false.
endif

end function run_test

end program
