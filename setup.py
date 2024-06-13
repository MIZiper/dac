from setuptools import setup, find_packages

setup(
    name="miz-dac",
    version="0.3.1",
    description="Data action context",
    long_description="Minimal measurement data analysis with nodes and container.",
    author="MIZiper",
    author_email="miziper@163.com",
    url="http://mizip.net/",
    download_url="https://github.com/MIZiper/dac.git",
    license="Apache-2.0",
    packages=find_packages(),
    package_data={"dac": ["plugins/*.yaml", "gui/resources/*.png"]},
    install_requires=["click", "numpy", "scipy", "matplotlib", "pyyaml", "nptdms"],
    extras_require={
        "GUI": ["pyqt5", "qscintilla"],
        "IPY": ["pyqt5", "qscintilla", "qtconsole"],
    },
    python_requires=">=3.10", # `|` used for types union
)
