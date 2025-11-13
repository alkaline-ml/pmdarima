python_sources = [
  '__init__.py',
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

py.extension_module(
  '_fourier',
  '_fourier.pyx',
  dependencies: _fourier_deps,
  install: true,
  subdir: 'pmdarima/preprocessing/exog'
)

subdir('tests')
