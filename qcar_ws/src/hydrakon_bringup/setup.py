import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hydrakon_bringup'

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
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Atlas Racing',
    maintainer_email='moizsaeed2004@gmail.com',
    description='Launch and node package for the QCar autonomous stack',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop = hydrakon_bringup.teleop:main',
            'slam = hydrakon_bringup.slam:main',
            'cone_fusion = hydrakon_bringup.cone_fusion_node:main',
            'ad_pure_pursuit = hydrakon_bringup.ad_pure_pursuit:main',
            'ad_acceleration = hydrakon_bringup.ad_acceleration:main',
            'adsdv_pure_pursuit = hydrakon_bringup.adsdv_pure_pursuit:main',
            'pure_pursuit = hydrakon_bringup.pure_pursuit:main',
            'sf_pure_pursuit = hydrakon_bringup.sf_pure_pursuit:main',
            'ap_pure_pursuit = hydrakon_bringup.ap_pure_pursuit:main',
            'lap_manager = hydrakon_bringup.lap_manager:main',
            'mo_slam = hydrakon_bringup.mo_slam:main',
            'fast_slam = hydrakon_bringup.fast_slamnode1:main',
            'interpolate_cones = hydrakon_bringup.interpolate_cones:main',
            'state_monitor = hydrakon_bringup.state_monitor:main',
            'odom_tf_bridge = hydrakon_bringup.odom_tf_bridge:main',
        ],
    },
)
