from ray.rllib.models.catalog import ModelCatalog
from attention_study.model.altr_rllib import AltrPolicy
from attention_study.model.graph_transformer_rllib import GraphTransformerPolicy

# register our model (put in an __init__ file later)
# https://docs.ray.io/en/latest/rllib-models.html#customizing-preprocessors-and-models
ModelCatalog.register_custom_model("altr_policy", AltrPolicy)
ModelCatalog.register_custom_model("graph_transformer_policy", AltrPolicy)