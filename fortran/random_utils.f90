module random_utils

use, intrinsic:: iso_fortran_env, only: wp=>real64

implicit none

real(wp),parameter :: pi = 4._wp*atan(1._wp)

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

end module random_utils