from setuptools import setup, find_packages

setup(
    name='StatArb_MS',
    version='0.0.1',
    description='Statistical Arbitrage strategy based on the work of Avellaneda and Lee (2008). The implementation allows for a new layer of complexity: filter a possible dynamics of the autoregressive coefficient in the AR(1) process by mean of GAS models.',
    url='https://github.com/DanieleMDiNosse/StatArb_MS.git',
    author='Daniele Maria Di Nosse',
    author_email='danielemdinosse@gmail.com',
    license='gnu general public license',
    packages = find_packages(),
    install_requires=['numpy', 'requests', 'scikit-learn', 'scipy', 'statsmodels', 'cython', 'pandas'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
