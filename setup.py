from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='jaxid',
    version='0.2',
    url='https://github.com/forgi86/jax-ident',
    author='Marco Forgione',
    author_email='marco.forgione1986@gmail.com',
    description='A code base for system identification with Jax.',
    packages=["jaxid"],
    install_requires=requirements,
    extras_require={
        'download datasets': ["requests", "googledrivedownloader"],
        'open datasets': ["pandas"],
        'generate plots': ['matplotlib'],
        'generate documentation': ["sphinx"]
    }
)