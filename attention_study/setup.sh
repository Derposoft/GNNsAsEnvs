# script to download/set up dependencies for this example

# 1. python libraries we use
pip3 install -r requirements.txt

# 2. pull submodules
git submodule update --init --recursive

# 3. install library dependencies
cd libraries/
pip3 install -e .

# garbage zone
#git clone https://github.com/Hanjun-Dai/pytorch_structure2vec.git libraries/s2v