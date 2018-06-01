from setuptools import setup

setup(
    name='cappresse',
    version='b1',
    packages=['testing.pyDAE', 'testing.old_tests', 'nmpc_mhe', 'nmpc_mhe.aux', 'nmpc_mhe.dync', 'nmpc_mhe.pyomo_dae',
              'snapshots', 'sample_mods', 'testing',
              'sample_mods.bfb', 'sample_mods.distl', 'sample_mods.distc_pyDAE', 'sample_mods.cstr_rodrigo'],
    url='',
    license='BSD-3',
    author='David Thierry',
    author_email='',
    description='The Control-Automated-Pyomo-Predictive-Sensitivity-State-Estimator '
)
