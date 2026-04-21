"""Exponential Inference configuration — extends base model config with routing params."""
from transformers import PretrainedConfig, AutoConfig


class ExponentialConfig(PretrainedConfig):
    model_type = "exponential"

    def __init__(
        self,
        base_model_id="Qwen/Qwen3-14B",
        default_active_heads_frac=0.5,
        default_exit_layer_frac=1.0,
        min_heads=4,
        local_kv_window=32,
        routing_method="sharpness",  # "sharpness", "entropy", "learned"
        **kwargs,
    ):
        self.base_model_id = base_model_id
        self.default_active_heads_frac = default_active_heads_frac
        self.default_exit_layer_frac = default_exit_layer_frac
        self.min_heads = min_heads
        self.local_kv_window = local_kv_window
        self.routing_method = routing_method

        # Load base config
        base_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
        for k, v in base_config.to_dict().items():
            if not hasattr(self, k):
                setattr(self, k, v)

        super().__init__(**kwargs)
