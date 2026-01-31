from setuptools import find_packages, setup
import os


def _read_version():
    env_version = os.getenv("AKG_CLI_VERSION")
    if env_version:
        return env_version
    try:
        with open("version.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.1.0"


def _read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
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
        "Sources": "https://gitee.com/mindspore/akg",
        "Issue Tracker": "https://gitee.com/mindspore/akg/issues",
    },
    license="Apache-2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[f"ai-kernel-generator=={version}"],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "akg_cli=ai_kernel_generator.cli.cli:app",
            "akg-cli=ai_kernel_generator.cli.cli:app",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
)
