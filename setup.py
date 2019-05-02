from setuptools import setup, find_packages

setup(name='foundry_hyperspectral_utils',
      version='0.0.1',
      description='Classes and functions for working with hyperspectral data',
      long_description=open('README.md', 'r').read(),
      # Author details
      author='Chris Chen',
      author_email='christopherchen@lbl.gov',
      # Choose your license
      license='BSD',
      # packages=['ScopeFoundry',
      #           'ScopeFoundry.scanning',
      #           'ScopeFoundry.examples',],
      # package_dir={'ScopeFoundry': './ScopeFoundry'},
      packages=find_packages(
          '.', exclude=['contrib', 'docs', 'tests', 'notebooks']),
      # include_package_data=True,
      )
