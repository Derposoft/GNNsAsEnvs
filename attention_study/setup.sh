# script to download/set up dependencies for this example

# 1. python libraries we use
# TODO update torch/pyG from torch==1.8 to torch==1.9 or 1.10; add +boost, +rdkit (?? if we end up using these?)
# we're using torch 1.8 for cpu. change this to gpu by running the command given at https://pytorch.org/get-started/locally/ if gpu is desired.
pip3 install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1+cu102
# we use pyG for gnn/gcns. from https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-geometric
pip install torch-geometric-temporal
# we use rllib! see: https://docs.ray.io/en/latest/installation.html
pip install -U ray[rllib]

# 2. github repos that we use
mkdir libraries/
# 2.1 clone and set up attention_routing repository
git clone https://github.com/wouterkool/attention-learn-to-route.git libraries/attention_routing
#cp setup/setup_package.py gnn_libraries/attention_routing/setup_package.py
#cd gnn_libraries/attention_routing
#python3 setup_package.py attention_routing https://github.com/wouterkool/attention-learn-to-route.git "numpy,scipy,torch>=1.7,tqdm,tensorboard_logger,matplotlib"
cp setup/setup_attention_routing.py libraries/setup.py
cd libraries/
pip3 install -e .

# 2.2 unused rn
git clone https://github.com/Hanjun-Dai/pytorch_structure2vec.git libraries/s2v