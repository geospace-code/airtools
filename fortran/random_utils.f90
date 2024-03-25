module random_utils

use, intrinsic:: iso_fortran_env, only: wp=>real64

implicit none

real(wp),parameter :: pi = 4 * atan(1._wp)

contains

impure elemental subroutine randn(noise)
! implements Box-Muller Transform
! https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
!
! Output:
! noise: Gaussian noise vector

real(wp),intent(out) :: noise
real(wp) :: u1, u2

call random_number(u1)
call random_number(u2)

noise = sqrt(-2._wp * log(u1)) * cos(2._wp * pi * u2)

end subroutine randn

end module random_utils
