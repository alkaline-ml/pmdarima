py.install_sources(
  ['__init__.py'],
  subdir: 'pmdarima/__check_build'
)

py.extension_module(
  '_check_build',
  '_check_build.pyx',
  dependencies: [py_dep, np_dep],
  install: true,
  subdir: 'pmdarima/__check_build'
)

subdir('tests')
