try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name = 'tanuki',
    packages = ['tanuki'],
    install_requires=['numpy', 'scikit-learn'],
    version = '0.1.0',
    description = 'A library for bandits',
    author = 'Todd Young',
    author_email = 'young.todd.mk@gmail.com',
    url = 'https://github.com/yngtodd/tanuki',
    keywords = ['reinforcement learning', 'bandits'],
)

