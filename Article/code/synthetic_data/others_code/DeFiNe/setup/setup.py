from setuptools import setup

setup(name='DeFiNe',
      version='0.1',
      description='De(composing) Fi(lamentous) Ne(tworks)',
      author='David Breuer <david.breuer@posteo.de>',
      url='http://mathbiol.mpimp-golm.mpg.de/DeFiNe/index.html',
      packages=['DeFiNe'],
      include_package_data=True,
      scripts=[
          'bin/run_DeFiNe.py',
      ],
      python_requires='>=2.*',
)
