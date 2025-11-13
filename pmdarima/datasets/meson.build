python_sources = [
  '__init__.py',
  '_base.py',
  'airpassengers.py',
  'ausbeer.py',
  'austres.py',
  'gasoline.py',
  'heartrate.py',
  'lynx.py',
  'stocks.py',
  'sunspots.py',
  'taylor.py',
  'wineind.py',
  'woolyrnq.py',
]

py.install_sources(
  python_sources,
  subdir: 'pmdarima/datasets'
)

# Install data files
install_subdir('data', install_dir: py.get_install_dir() / 'pmdarima/datasets', strip_directory: false)

subdir('tests')
