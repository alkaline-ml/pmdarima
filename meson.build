project(
  'pmdarima',
  'c', 'cython',
  version: '2.1.1dev0',
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
np_dep = dependency('numpy', required: true)

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
