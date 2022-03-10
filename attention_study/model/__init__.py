from ray.rllib.models.catalog import ModelCatalog
#from attention_study.model.altr_rllib import AltrPolicy
from attention_study.model.graph_transformer_rllib import GraphTransformerPolicy
from attention_study.model.fc_rllib import FCPolicy

# register our model (put in an __init__ file later)
# https://docs.ray.io/en/latest/rllib-models.html#customizing-preprocessors-and-models
#ModelCatalog.register_custom_model("altr_policy", AltrPolicy)
ModelCatalog.register_custom_model("graph_transformer_policy", GraphTransformerPolicy)
ModelCatalog.register_custom_model("fc_policy", FCPolicy)