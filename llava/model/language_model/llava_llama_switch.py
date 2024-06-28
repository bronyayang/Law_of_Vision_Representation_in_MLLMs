from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from .llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel




class SwitchConfig(LlamaConfig):
    model_type = "llava_switch"


class LlavaLlamaSwtichForCausalLM(LlavaLlamaForCausalLM):
    config_class = SwitchConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.W = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sigma = 0
        # self.W_dummy = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        postive:torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        # c+εWc
        if not (postive is None):
            postive = postive.unsqueeze(1).unsqueeze(2)
            tensor_type = hidden_states.dtype
            hidden_states = hidden_states +  postive * (self.W(hidden_states))
            hidden_states = hidden_states.to(tensor_type)
        elif self.sigma:
            tensor_type = hidden_states.dtype
            hidden_states = hidden_states +  self.sigma * (self.W(hidden_states))
            hidden_states = hidden_states.to(tensor_type)


        # if postive:
        #     hidden_states = hidden_states + sigma * (self.W(hidden_states) + self.W_dummy(hidden_states))
        # else:
        #     hidden_states = hidden_states + sigma * (self.W(hidden_states) + self.W_dummy(hidden_states))

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

