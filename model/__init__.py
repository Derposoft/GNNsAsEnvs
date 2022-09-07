from ray.rllib.models.catalog import ModelCatalog
from model.gat_rllib import GNNPolicy
from model.graph_transformer_rllib import GraphTransformerPolicy
from model.hybrid_rllib import HybridPolicy
from model.fc_rllib import FCPolicy

# register our model (put in an __init__ file later)
# https://docs.ray.io/en/latest/rllib-models.html#customizing-preprocessors-and-models
ModelCatalog.register_custom_model("gnn_policy", GNNPolicy)
ModelCatalog.register_custom_model("graph_transformer_policy", GraphTransformerPolicy)
ModelCatalog.register_custom_model("hybrid_policy", HybridPolicy)
ModelCatalog.register_custom_model("fc_policy", FCPolicy)

"""
not under development: "attention learn to route"-based model
"""
#from model.altr_rllib import AltrPolicy
#ModelCatalog.register_custom_model("altr_policy", AltrPolicy)