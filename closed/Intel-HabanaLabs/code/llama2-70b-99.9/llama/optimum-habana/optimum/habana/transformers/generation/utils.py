# coding=utf-8
# Copyright 2022 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.autograd.profiler as profiler
import torch.distributed as dist
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    GenerateOutput,
    GenerationMixin,
    GenerationMode,
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    GreedySearchOutput,
)
from transformers.generation import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ModelOutput


from ...utils import HabanaProfile
from ..integrations.deepspeed import unwrap_deepspeed_model

MODELS_OPTIMIZED_WITH_STATIC_SHAPES = [
    "bloom",
    "gpt2",
    "opt",
    "gptj",
    "gpt_neox",
    "llama",
    "falcon",
    "codegen",
    "gpt_bigcode",
    "bart",
    "mpt",
    "t5",
    "mistral",
]

class GaudiGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trim_logits = kwargs.get("trim_logits", None)
        self.static_shapes = kwargs.get("static_shapes", None)
        self.ignore_eos = kwargs.get("ignore_eos", None)
        self.attn_softmax_bf16 = kwargs.get("attn_softmax_bf16", None)
        self.limit_hpu_graphs = kwargs.get("limit_hpu_graphs", None)
        self.reuse_cache = kwargs.get("reuse_cache", None)
        self.bucket_size = kwargs.get("bucket_size", -1)


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from .streamers import BaseStreamer



def incrementor(bucket_size, prompt_len):
    assert bucket_size > 0
    passnum = -1
    while True:
        passnum += 1
        if passnum == 0:
            token_idx = prompt_len
            allocated_space = int(math.ceil(prompt_len / bucket_size) * bucket_size)
            need_expansion = not (prompt_len == allocated_space)
        else:
            token_idx += 1
            need_expansion = token_idx >= allocated_space
            if need_expansion:
                assert (allocated_space - token_idx) <= bucket_size
                allocated_space += bucket_size
        yield {
            "allocated_space": allocated_space,
            "passnum": passnum,
            "token_idx": token_idx,
            "need_expansion": need_expansion,
        }


class StaticMaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.cur_step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.cur_step += 1
        return self.cur_step >= self.max_steps


class GaudiGenerationMixin(GenerationMixin):
    def _get_hpu_graphs_kwargs(self, model_kwargs):
        hpu_graphs_kwargs = {}
        if model_kwargs["limit_hpu_graphs"]:
            hpu_graphs_kwargs.update({"bypass_hpu_graphs": False})
            if "first_token" not in model_kwargs.keys():
                model_kwargs["first_token"] = True
                hpu_graphs_kwargs.update({"bypass_hpu_graphs": True})
        return hpu_graphs_kwargs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # mark to identify starting from second token
        model_kwargs["first_token"] = False
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        token_idx = model_kwargs.get("token_idx", None)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                if token_idx is not None:
                    attention_mask.index_fill_(1, token_idx, 1)
                else:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                model_kwargs["attention_mask"] = attention_mask
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                if token_idx is not None:
                    decoder_attention_mask.index_fill_(1, token_idx, 1)
                else:
                    decoder_attention_mask = torch.cat(
                        [
                            decoder_attention_mask,
                            decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1)),
                        ],
                        dim=-1,
                    )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        if token_idx is not None:
            token_idx.add_(1)

        return model_kwargs


    def _prepare_decoder_attention_mask(
        self,
        max_steps: int,  # current stopping criteria
        batch_size: int,
        pad_token_id: int,
        device: str,
        dtype: str = bool,
    ) -> torch.Tensor:
        x = torch.zeros((batch_size, max_steps), device=device, dtype=dtype)
        return x.index_fill(1, torch.tensor([0]), 1)  # First the position with pad_token_id


    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.

        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        token_idx = model_kwargs.get("token_idx", None)

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        if token_idx is None:
            decoder_input_ids_start = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )
        else:
            # creating padded decoder_input_ids to achieve static shapes. Later new tokens once generated are copied in to decoder_input_ids based on token_idx
            decoder_input_ids_start = (
                torch.ones((batch_size, self.generation_config.max_length), dtype=torch.long, device=device)
                * decoder_start_token_id
            )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask
        return decoder_input_ids, model_kwargs

    def _get_stopping_criteria(
        self, generation_config: GaudiGenerationConfig, stopping_criteria: Optional[StoppingCriteriaList]
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            if (
                generation_config.static_shapes
                and self.config.is_encoder_decoder
                and (
                    self.generation_config.generation_mode == GenerationMode.GREEDY_SEARCH
                    or self.generation_config.generation_mode == GenerationMode.BEAM_SEARCH
                )
            ):
                criteria.append(StaticMaxLengthCriteria(generation_config.max_length))
            else:
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                criteria.append(
                    MaxLengthCriteria(
                        max_length=generation_config.max_length,
                        max_position_embeddings=max_position_embeddings,
                    )
                )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    @torch.no_grad()
    def update_model_kwargs_for_bucketing(self, params, input_ids, model_kwargs, pad_token_id, bucket_size):
        if params["need_expansion"]:
            # Pad inputs to have static shapes during generation, this gives better performance than dynamic shapes on HPUs
            pad_amount = params["allocated_space"] - input_ids.shape[-1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_amount), value=pad_token_id)
            if model_kwargs["attention_mask"] is not None:
                model_kwargs["attention_mask"] = torch.nn.functional.pad(
                    model_kwargs["attention_mask"], (0, pad_amount), value=0
                )
            else:
                assert False, "Not tested for cases where attn_mask isnt passed"

            if "past_key_values" in model_kwargs:

                def create_pad_arg(pad_amount, i, j):
                    if model_kwargs["past_key_values"][0][0].dim() == 3:
                        assert self.config.model_type == "bloom"
                        if j == 0:
                            return (0, pad_amount)
                        elif j == 1:
                            return (0, 0, 0, pad_amount)
                        else:
                            assert False
                    elif model_kwargs["past_key_values"][0][0].dim() == 4:
                        return (0, 0, 0, pad_amount)  # llama, falcon
                    else:
                        assert False, "Unknown case, please handle, or dont use bucketing"

                new_kv = [None for i in range(len(model_kwargs["past_key_values"]))]
                for i in range(len(model_kwargs["past_key_values"])):
                    tmp_lst = [None for j in range(len(model_kwargs["past_key_values"][i]))]
                    for j in range(len(model_kwargs["past_key_values"][i])):
                        pad_tuple = create_pad_arg(pad_amount, i, j)
                        # Different models might have different shapes of kv-cache
                        # create_pad_arg handles them on a per-model basis
                        # This is a necessary (but not sufficient) condition: what ever dimension we are padding, should be a multiple of bucket_size
                        # This check is added in case we get a new model with a new kv-cache structure, and we attempt to pad some wrong dimension
                        assert model_kwargs["past_key_values"][i][j].shape[-(len(pad_tuple) // 2)] % bucket_size == 0
                        tmp_lst[j] = torch.nn.functional.pad(
                            model_kwargs["past_key_values"][i][j], pad_tuple, value=pad_token_id
                        )
                    new_kv[i] = tuple(tmp_lst)
                model_kwargs["past_key_values"] = tuple(new_kv)

        if "token_idx" not in model_kwargs:
            model_kwargs["token_idx"] = torch.tensor(params["token_idx"], device=self.device)
        return input_ids, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GaudiGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        lazy_mode: Optional[bool] = False,
        hpu_graphs: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        if hpu_graphs and not lazy_mode:
            raise ValueError(
                "`hpu_graphs` is True but `lazy_mode` is False. HPU graphs require `lazy_mode` to be set to True."
            )

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                self.generation_config
            ):
                new_generation_config = GaudiGenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        if generation_config.static_shapes is None:
            generation_config.static_shapes = True
        if generation_config.ignore_eos is None:
            generation_config.ignore_eos = kwargs.get("ignore_eos", lazy_mode)
        generation_config.validate()
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        self._validate_model_kwargs(model_kwargs.copy())
        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{generation_config.eos_token_id} for open-end generation."
            )
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        is_greedy_or_beam_and_bucket = generation_config.bucket_size > 0 and (
            self._get_generation_mode(generation_config, assistant_model) == GenerationMode.GREEDY_SEARCH
            or self._get_generation_mode(generation_config, assistant_model) == GenerationMode.BEAM_SEARCH
        )
        model_kwargs["bucket_size"] = generation_config.bucket_size if generation_config.static_shapes else -1
        if generation_config.reuse_cache:
            assert generation_config.bucket_size <= 0, "reuse_cache and bucketing flags set together"

        if generation_config.static_shapes:
            # Pad inputs to have static shapes during generation, this gives better performance than dynamic shapes on HPUs
            # In encoder_decoder models, Inputs are already padded

            assert not self.config.is_encoder_decoder, "Encoder-decoder not supported"
            # only pad if bucket_size < -1. If we are bucketing (bucket_size > 0), then that is taken care in greedy_search()
            if not is_greedy_or_beam_and_bucket:
                # token_idx is the current index in the generation process, it is incremented each time a new token is generated
                token_idx = inputs_tensor.shape[-1]
                model_kwargs["token_idx"] = torch.tensor(token_idx, device=inputs_tensor.device)
                inputs_tensor = torch.nn.functional.pad(
                    inputs_tensor, (0, generation_config.max_new_tokens), value=generation_config.pad_token_id
                )
                for other_inputs in ["attention_mask", "token_type_ids"]:
                    if model_kwargs.get(other_inputs) is not None:
                        model_kwargs[other_inputs] = torch.nn.functional.pad(
                            model_kwargs[other_inputs], (0, generation_config.max_new_tokens), value=0
                        )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if generation_config.pad_token_id is not None:
                position = model_kwargs["token_idx"] - 1 if "token_idx" in model_kwargs else -1
                if (
                    len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, position] == generation_config.pad_token_id) > 0
                ):
                    logger.warning(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
        # 5. Prepare `input_ids` which will be used for auto-regressive generation

        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # determine whether introduce trim_logits feature
        model_kwargs["trim_logits"] = generation_config.trim_logits

        # determine whether attention softmax needs to execute in lower precision
        model_kwargs["attn_softmax_bf16"] = generation_config.attn_softmax_bf16

        # determine whether limit_hpu_graphs needs to be used
        model_kwargs["limit_hpu_graphs"] = generation_config.limit_hpu_graphs

        # prepare for allocate kv cache
        model_kwargs["reuse_cache"] = generation_config.reuse_cache

        if not self.config.is_encoder_decoder:
            calculated_max_length = input_ids.shape[-1]
            if not generation_config.static_shapes and generation_config.max_new_tokens is not None:
                calculated_max_length = input_ids.shape[-1] + generation_config.max_new_tokens
            if generation_config.use_cache and generation_config.reuse_cache:
                bs, _ = input_ids.shape
                if not is_greedy_or_beam_and_bucket:
                    unwrap_deepspeed_model(self).allocate_kv_cache(
                        bs * generation_config.num_beams,
                        calculated_max_length,
                        token_idx,
                    )
            if self.config.model_type in ["llama"]:
                if self.config.max_position_embeddings < calculated_max_length:
                    unwrap_deepspeed_model(self).update_sincos_cache(seq_len=calculated_max_length)

        # 7. determine generation mode
        generation_mode = self._get_generation_mode(generation_config, assistant_model)
        if generation_config.bucket_size > 0:
            assert generation_config.static_shapes, "bucket_size > 0 can be set only when static_shapes is set"
        # if generation_config.bucket_size <= 0, padding is handled by the generating fn (like greedy_search)
        if generation_config.static_shapes and generation_config.bucket_size > 0:
            assert (
                generation_mode == GenerationMode.GREEDY_SEARCH or generation_mode == GenerationMode.BEAM_SEARCH
            ), "generation_config.bucket_size > 0 supported only for greedy mode"

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                (
                    "You are calling .generate() with the `input_ids` being on a device type different"
                    f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                    f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                    " Please make sure that you have put `input_ids` to the"
                    f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                    " running `.generate()`."
                ),
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        self.generation_config.generation_mode = generation_mode
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        if "token_idx" in model_kwargs and not self.config.is_encoder_decoder:
            if generation_config.max_new_tokens is not None:
                stopping_criteria.append(StaticMaxLengthCriteria(generation_config.max_new_tokens))
            else:
                raise ValueError(
                    "You need to set `max_new_tokens` in your generation configuration to use static shapes."
                )

        if generation_config.static_shapes and generation_config.bucket_size > 0:
            stopping_criteria = StoppingCriteriaList(
                [
                    StaticMaxLengthCriteria(generation_config.max_new_tokens)
                    if type(crit) == MaxLengthCriteria
                    else crit
                    for crit in stopping_criteria
                ]
            )

        # In lazy mode, import Habana torch to be able to add mark_step()
        if lazy_mode:
            import habana_frameworks.torch.core as htcore

            self.htcore_generation = htcore

        # 10. go into different generation modes
        with profiler.record_function("greedy_search"):
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                lazy_mode=lazy_mode,
                ignore_eos=generation_config.ignore_eos,
                profiling_warmup_steps=profiling_warmup_steps,
                profiling_steps=profiling_steps,
                **model_kwargs,
            )


    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        lazy_mode: Optional[bool] = False,
        ignore_eos: Optional[bool] = False,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                (
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
                ),
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        if not ignore_eos:
            unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        hb_profer = HabanaProfile(warmup=profiling_warmup_steps, active=profiling_steps)
        hb_profer.start()
        this_peer_finished = False  # used by synced_gpus only
        bucket_size = model_kwargs["bucket_size"]

        prompt_len = input_ids.shape[-1]
        if bucket_size >= 0:
            inc = iter(incrementor(bucket_size, prompt_len))
        if bucket_size > 0:
            assert "position_ids" not in model_kwargs, "Untested path"

        while True:
            if lazy_mode:
                self.htcore_generation.mark_step()

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            if bucket_size > 0:
                # it will not have been padded if bucket_size > 0
                params = next(inc)
                input_ids, model_kwargs = self.update_model_kwargs_for_bucketing(
                    params, input_ids, model_kwargs, pad_token_id, bucket_size
                )

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            hpu_graphs_kwargs = self._get_hpu_graphs_kwargs(model_kwargs)

            with profiler.record_function("forward"):
                # forward pass to get next token
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **hpu_graphs_kwargs,
                )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            token_idx = model_kwargs.get("token_idx", None)
            if token_idx is not None and outputs.logits.shape[-2] > 1:
                # case1 (w/o KV caching): outputs.logits.shape: [batch_size, max_length, vocab_size]
                if self.config.is_encoder_decoder:
                    next_token_logits = outputs.logits[:, token_idx - 1, :]
                    next_tokens_scores = logits_processor(input_ids[:, :token_idx], next_token_logits)
                else:
                    next_token_logits = torch.index_select(outputs.logits, -2, token_idx - 1).squeeze(-2)
                    next_tokens_scores = logits_processor(input_ids, next_token_logits)
            else:
                next_token_logits = outputs.logits[:, -1, :]
                if token_idx is not None and self.config.is_encoder_decoder:
                    # case2 (with KV caching): outputs.logits.shape: [batch_size, 1, vocab_size]
                    next_tokens_scores = logits_processor(input_ids[:, :token_idx], next_token_logits)
                else:
                    # case3 (default case): token_idx is None
                    next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # finished sentences should have their next token be a padding token
            if not ignore_eos and eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            if token_idx is not None:
                input_ids.index_copy_(
                    1, token_idx, next_tokens.unsqueeze(-1) if next_tokens.dim() == 1 else next_tokens
                )
            else:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if not ignore_eos and eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
                # stop when each sentence is finished
                if not ignore_eos and unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            hb_profer.step()

            if this_peer_finished and not synced_gpus:
                break

        hb_profer.stop()
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids