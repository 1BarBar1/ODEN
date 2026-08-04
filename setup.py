from setuptools import setup

package_name = 'oden'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
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
