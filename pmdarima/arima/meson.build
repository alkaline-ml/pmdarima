python_sources = [
  '__init__.py',
  '_auto_solvers.py',
  '_context.py',
  '_doc.py',
  '_validation.py',
  'approx.py',
  'arima.py',
  'auto.py',
  'seasonality.py',
  'stationarity.py',
  'utils.py',
]

py.install_sources(
  python_sources,
  subdir: 'pmdarima/arima'
)

# ARIMA Cython extension with BLAS
_arima_deps = [py_dep, np_dep]
if blas_dep.found()
  _arima_deps += [blas_dep]
endif
if m_dep.found()
  _arima_deps += [m_dep]
endif

py.extension_module(
  '_arima',
  '_arima.pyx',
  dependencies: _arima_deps,
  install: true,
  subdir: 'pmdarima/arima'
)

subdir('tests')
