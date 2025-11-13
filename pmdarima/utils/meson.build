python_sources = [
  '__init__.py',
  '_show_versions.py',
  'array.py',
  'metaestimators.py',
  'visualization.py',
  'wrapped.py',
]

py.install_sources(
  python_sources,
  subdir: 'pmdarima/utils'
)

# Array Cython extension with BLAS
_array_deps = [py_dep, np_dep]
if blas_dep.found()
  _array_deps += [blas_dep]
endif
if m_dep.found()
  _array_deps += [m_dep]
endif

py.extension_module(
  '_array',
  '_array.pyx',
  dependencies: _array_deps,
  install: true,
  subdir: 'pmdarima/utils'
)

subdir('tests')
