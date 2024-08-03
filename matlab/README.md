# Instructions for MATLAB users
In order to call the Python functions from MATLAB, plase follow these intructions.
[here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

The code has been tested with Python 3.11 and MATLAB R2024a.

1. Install python on your computer
2. Create a virtual environment and install the required packages
```bash
cd PATH_TO_THE_REPO
pip install poetry
poetry install
source .venv/bin/activate
```
3. Set up the python environment in MATLAB as described [here](https://www.mathworks.com/help/matlab/ref/pyenv.html).
On Windows:
```matlab  
pyenv(Version="3.11")
pyenv('Executable','PATH_TO_THE_REPO/.venv/Scrips/python.exe')
pyenv(ExecutionMode="OutOfProcess")
```
On Linux:
```matlab
pyenv(Version="3.11")
pyenv('Executable','PATH_TO_THE_REPO/.venv/bin/python.exe')
pyenv(ExecutionMode="OutOfProcess")
```
4. Execute the code from the file matlab_test.m
