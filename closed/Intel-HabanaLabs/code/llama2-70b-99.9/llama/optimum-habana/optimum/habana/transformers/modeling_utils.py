def adapt_transformers_to_gaudi():
    import transformers
    from optimum.habana.transformers.generation.utils import GaudiGenerationConfig, GaudiGenerationMixin
    from optimum.habana.transformers.models.llama.modeling_llama import (
        GaudiLlamaAttention,
        GaudiLlamaDecoderLayer,
        GaudiLlamaForCausalLM,
        GaudiLlamaMLP,
        GaudiLlamaModel,
        gaudi_llama_rmsnorm_forward
    )
    transformers.generation.GenerationMixin.generate = GaudiGenerationMixin.generate
    transformers.generation.GenerationMixin._update_model_kwargs_for_generation = (
        GaudiGenerationMixin._update_model_kwargs_for_generation
    )
    transformers.generation.GenerationMixin.update_model_kwargs_for_bucketing = (
        GaudiGenerationMixin.update_model_kwargs_for_bucketing
    )
    transformers.generation.GenerationMixin._get_hpu_graphs_kwargs = GaudiGenerationMixin._get_hpu_graphs_kwargs
    transformers.generation.GenerationMixin._expand_inputs_for_generation = staticmethod(
        GaudiGenerationMixin._expand_inputs_for_generation
    )
    transformers.generation.GenerationMixin._prepare_attention_mask_for_generation = (
        GaudiGenerationMixin._prepare_attention_mask_for_generation
    )
    transformers.generation.GenerationMixin._prepare_decoder_input_ids_for_generation = (
        GaudiGenerationMixin._prepare_decoder_input_ids_for_generation
    )
    transformers.generation.GenerationMixin._get_stopping_criteria = GaudiGenerationMixin._get_stopping_criteria
    transformers.generation.GenerationMixin._prepare_decoder_attention_mask = (
        GaudiGenerationMixin._prepare_decoder_attention_mask
    )
    transformers.generation.GenerationMixin._validate_model_kwargs = GaudiGenerationMixin._validate_model_kwargs
    transformers.generation.GenerationMixin.greedy_search = GaudiGenerationMixin.greedy_search
    transformers.generation.GenerationConfig = GaudiGenerationConfig
    transformers.modeling_utils.GenerationConfig = GaudiGenerationConfig
    transformers.models.llama.modeling_llama.LlamaForCausalLM = GaudiLlamaForCausalLM
    transformers.models.llama.modeling_llama.LlamaModel = GaudiLlamaModel
    transformers.models.llama.modeling_llama.LlamaAttention = GaudiLlamaAttention
    transformers.models.llama.modeling_llama.LlamaMLP = GaudiLlamaMLP
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = GaudiLlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = gaudi_llama_rmsnorm_forward