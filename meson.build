project(
  'pmdarima',
  'c', 'cython',
  version: run_command(['build_tools/get_tag.py'], check: true).stdout().strip(),
  license: 'MIT',
  meson_version: '>= 1.1.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Get numpy dependency
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

# Add Cython as a dependency
cython = find_program('cython')

# Find BLAS/LAPACK
blas_dep = dependency('openblas', required: false)
if not blas_dep.found()
  blas_dep = dependency('blas', required: false)
endif
if not blas_dep.found()
  # Fallback for macOS Accelerate framework
  blas_dep = dependency('appleframeworks', modules: ['Accelerate'], required: false)
endif

# C math library
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

# Process subdirectories
subdir('pmdarima')
