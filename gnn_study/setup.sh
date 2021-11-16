# script to download/set up dependencies for this example

# 1. python libraries we use
# TODO update torch/pyG from torch==1.8 to torch==1.9 or 1.10; add +boost, +rdkit (?? if we end up using these?)
# we're using torch 1.8 for cpu. change this to gpu by running the command given at https://pytorch.org/get-started/locally/ if gpu is desired.
pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
# we use pyG for gnn/gcns. from https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/installation.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-geometric
pip install torch-geometric-temporal
# we use rllib! see: https://docs.ray.io/en/latest/installation.html
pip install -U ray[rllib]

# 2. github repos that we use
mkdir gnn_libraries/
git clone https://github.com/Hanjun-Dai/pytorch_structure2vec.git gnn_libraries/s2v