from setuptools import setup

setup(name='ot_class',
      version='0.1',
      description='Optimal transport methods for classification',
      url='https://github.com/gstamatiou/optimal-classification',
      author='Giorgos Stamatiou',
      author_email='s6gestam@uni-bonn.de',
      license='MIT',
      packages=['ot_class'],
      install_requires=[
          'graphlearning',
          'numpy',
          'matplotlib',
          'Pillow'
          ],
      zip_safe=False)
