import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hydrakon_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Atlas Racing',
    maintainer_email='moizsaeed2004@gmail.com',
    description='Camera perception nodes for the QCar autonomous stack',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cone_locator = hydrakon_camera.cone_locator:main',
        ],
    },
)
