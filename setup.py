from setuptools import setup

setup(
    name='nmpc_mhe_q',
    version='0.1',
    packages=['nmpc_mhe', 'nmpc_mhe.aux', 'nmpc_mhe.dync', 'snapshots', 'sample_mods', 'sample_mods.bfb',
              'sample_mods.distl', 'nmpc_mhe.pyomo_dae', 'sample_mods.cstr_rodrigo'],
    url='https://github.com/dthierry/nmpc_mhe_q',
    license='MIT',
    author='David Thierry',
    author_email='dmolinat@andrew.cmu.edu',
    description='Toolbox for NMPC and MHE'
)
