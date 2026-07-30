from setuptools import setup

package_name = 'oden'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Harry Grehag',
    maintainer_email='harygrehag@gmail.com',
    description='Object detection and Estimation Nodes',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'publisher = oden.processing_node:main',

        ],
    },
)
