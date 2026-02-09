from setuptools import find_packages, setup
import os


_HERE = os.path.abspath(os.path.dirname(__file__))


def _read_version():
    env_version = os.getenv("AKG_CLI_VERSION")
    if env_version:
        return env_version
    try:
        with open(os.path.join(_HERE, "version.txt"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.2.0"


def _read_readme():
    with open(os.path.join(_HERE, "README.md"), "r", encoding="utf-8") as fh:
        return fh.read()


version = _read_version()

setup(
    name="akg-cli",
    version=version,
    author="The MindSpore Authors",
    author_email="contact@mindspore.cn",
    description="AKG CLI wrapper package",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="https://www.mindspore.cn/",
    project_urls={
        "Sources": "https://gitcode.com/mindspore/akg/tree/br_akg_agents/akg_agents/akg-cli",
        "Issue Tracker": "https://gitcode.com/mindspore/akg/issues",
    },
    license="Apache-2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[f"akg_agents=={version}"],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "akg_cli=akg_agents.cli.cli:app",
            "akg-cli=akg_agents.cli.cli:app",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
)
