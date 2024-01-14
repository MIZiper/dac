from setuptools import setup, find_packages

setup(
    name="dac",
    version="0.0.1",
    description="Data action context",
    long_description="Minimal measurement data analysis with nodes and container.",
    author="MIZiper",
    author_email="miziper@163.com",
    url="http://mizip.net/",
    download_url="https://github.com/MIZiper/dac.git",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=["click", "numpy", "scipy", "pyqt5", "matplotlib", "pyyaml", "qscintilla", "nptdms"],
    requires=["qtconsole"],
    python_requires=">=3.10", # `|` used for types union
)