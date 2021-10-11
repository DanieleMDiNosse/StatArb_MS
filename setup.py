from setuptools import setup, find_packages

setup(
    name='StatArb_MS',
    version='0.0.1',
    description='Statistical Arbitrage tool',
    url='https://github.com/DanieleMDiNosse/StatArb_MS.git',
    author='Di Nosse Daniele Maria',
    author_email='danielemdinosse@gmail.com',
    license='gnu general public license',
    packages = find_packages(),
    install_requires=['numpy', 'requests', 'scikit-learn'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)
