! solve b=Ax using parallel log-ent mart.
! Matlab by Joshua Semeter
! Fortran by Michael Hirsch

use, intrinsic:: iso_fortran_env, only: stderr=>error_unit
use art, only: logmart
use random_utils, only: init_random_seed, randn, wp

implicit none

integer, parameter :: N=3
real(wp) :: A(N,N)
real(wp), parameter :: x_true(N)=[5,5,5]
real(wp), parameter :: errtol=0.05_wp
real(wp), dimension(N) :: x, noise, b,bias
logical :: add_bias, add_noise

add_bias = .false.
add_noise = .true.

call init_random_seed()

A = reshape([1,0,0, &
             0,1,0, &
             0,0,1], shape(A), order=[2,1])

block
integer :: i
print *, 'A ='
do i = 1,size(A,1)
  print '(3F10.3)', A(i,:)
enddo
end block

! ---- noisy observation
if (add_bias) then
  call randn(bias)
  bias = 0.01_wp * bias
  print '(/,A,3F10.3)','bias',bias
  A = A * spread(bias,2,N)
endif

if (add_noise) then
  call randn(noise)
  noise = 0.01_wp * noise
  print '(/,A,3F10.3)', 'noise',noise
  b = matmul(A,x_true) + noise
endif

! ---- inversion
call logmart(A,b,x=x)

! --- check estimate
if (any(abs(x-x_true) > errtol*maxval(x_true))) then
  print *,x
  write (stderr,*) 'larger than',errtol*100,' % error'
  stop 1
endif

print '(/,A)','OK: logmart'

end program