# bridge_pkg/setup.py

from setuptools import setup

package_name = 'bridge_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','pyzmq'],
    zip_safe=True,
    maintainer='autumnrtx',
    maintainer_email='ge26xif@mytum.de',
    description='A bridge node to forward ROS2 messages to Python 3.8 environment via ZeroMQ',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_node = bridge_pkg.bridge_node:main',
        ],
    },
)
