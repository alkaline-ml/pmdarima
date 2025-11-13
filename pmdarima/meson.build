# Install Python source files
py.install_sources(
  [
    '__init__.py',
    'base.py',
    'context_managers.py',
    'decorators.py',
    'metrics.py',
    'pipeline.py',
    'warnings.py',
  ],
  subdir: 'pmdarima'
)

# Process subdirectories with extensions
subdir('__check_build')
subdir('arima')
subdir('preprocessing')
subdir('utils')

# Install other Python subpackages
subdir('_build_utils')
subdir('compat')
subdir('datasets')
subdir('model_selection')
subdir('tests')
