from distutils.core import setup

setup(
    name='ramp_systems',
    package_dir={'':'src'},
    packages = ["ramp_systems","ramp_to_hill"],
    install_requires = ["DSGRN","sympy"],
    author="William Duncan",
    url='https://github.com/will-duncan/ramp_systems'
    )