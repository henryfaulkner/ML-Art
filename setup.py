from setuptools import setup

projectName = 'NeuralStyleTransfer'

setup(
    name=projectName,
    version='1.0',
    packages=[projectName],
    entry_points={
        'console_scripts': [
            projectName + " = " + projectName + ".__main__:main"
        ]
    },
    install_requires=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'pillow',
        'requests',
        'opencv-python'
    ]
)
