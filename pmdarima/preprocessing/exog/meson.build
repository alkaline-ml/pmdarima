python_sources = [
  '__init__.py',
  'base.py',
  'dates.py',
  'fourier.py',
]

py.install_sources(
  python_sources,
  subdir: 'pmdarima/preprocessing/exog'
)

# Fourier Cython extension with BLAS
_fourier_deps = [py_dep, np_dep]
if blas_dep.found()
  _fourier_deps += [blas_dep]
endif
if m_dep.found()
  _fourier_deps += [m_dep]
endif

_fourier_c_args = []
if host_machine.system() == 'windows'
  _fourier_c_args += ['-D_USE_MATH_DEFINES']
endif

py.extension_module(
  '_fourier',
  '_fourier.pyx',
  dependencies: _fourier_deps,
  c_args: _fourier_c_args,
  install: true,
  subdir: 'pmdarima/preprocessing/exog'
)

subdir('tests')
