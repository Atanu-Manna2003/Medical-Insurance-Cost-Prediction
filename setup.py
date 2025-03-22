from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns a list of requirements.
    '''
    requirements = []
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()  # Read all lines
        requirements = [req.strip() for req in requirements]  # Remove extra spaces and newlines

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)  # Remove '-e .'

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='atanu',
    author_email='atanumanna527@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)