# Install test files
python_sources = [
  '__init__.py',
  'test_approx.py',
  'test_arima.py',
  'test_arima_diagnostics.py',
  'test_auto.py',
  'test_auto_solvers.py',
  'test_c_arima.py',
  'test_context.py',
  'test_seasonality.py',
  'test_stationarity.py',
  'test_utils.py',
  'test_validation.py',
]

py.install_sources(
  python_sources,
  subdir: 'pmdarima/arima/tests'
)

install_subdir('data', install_dir: py.get_install_dir() / 'pmdarima/arima/tests', strip_directory: false)
