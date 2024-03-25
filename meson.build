project('airtools', 'fortran',
  default_options : ['buildtype=release'])


art = library('art', 'fortran/logmart.f90')

test_exe = executable('test_logmart',
  sources: ['fortran/test_logmart.f90', 'fortran/random_utils.f90'],
  link_with : art)

test('logmart', test_exe)
