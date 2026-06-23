from setuptools import find_packages, setup

package_name = 'asv_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/step3b_gazebo.launch.py',
            'launch/step4_ocean.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='svillhauer',
    maintainer_email='svillhauer@g.ucla.edu',
    description='ROS 2 bridge: EKF + obs builder for the two-glider ASV policy.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bridge_node      = asv_bridge.bridge_node:main',
            'policy_node      = asv_bridge.policy_node:main',
            'glider_viz_node  = asv_bridge.glider_viz_node:main',
        ],
    },
)
