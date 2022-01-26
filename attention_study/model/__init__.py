from ray.rllib.models.catalog import ModelCatalog
from attention_study.model.attention_policy import PolicyModel

# register our model (put in an __init__ file later)
# https://docs.ray.io/en/latest/rllib-models.html#customizing-preprocessors-and-models
ModelCatalog.register_custom_model("policy_model", PolicyModel)