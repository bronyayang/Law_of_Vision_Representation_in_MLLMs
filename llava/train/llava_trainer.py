import os
import torch
import warnings
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Mapping
import json
import sys

from torch.utils.data import Sampler

from transformers import Trainer, Seq2SeqTrainer, DataCollator, DataCollatorForSeq2Seq
from transformers.trainer import (
    has_length
)
from typing import List, Optional
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.deepspeed import is_deepspeed_zero3_enabled

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

class TrainerDifferentCollatorMixin:
    def __init__(self,
                 tokenizer,
                 *args,
                 train_collator: Optional[DataCollator] = None,
                 eval_collator: Optional[DataCollator] = None,
                 test_collator: Optional[DataCollator] = None,
                 **kwargs):
        if train_collator is None and eval_collator is None and test_collator is None:
            raise ValueError("use different collator for trainer but get no collator function.")
        if eval_collator is not None and test_collator is not None and eval_collator != test_collator:
            warnings.warn('[WARNING!!!] use different collator for eval and test. but maybe do_eval and '
                          'do_predict both use trainer.predict (i.e. only test_collator is used.) u should'
                          'check your code and know exactly what u are doing.')
        self._train_collator = train_collator
        self._eval_collator = eval_collator if eval_collator is not None else self._train_collator
        self._test_collator = test_collator if test_collator is not None else self._eval_collator
        if "data_collator" in kwargs and kwargs["data_collator"] is not None:
            warnings.warn("use different collator for trainer but get 'data_collator' argument. It will take no effect and be ignored.")
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_train_dataloader(self) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._train_collator
        dataloader = super().get_train_dataloader()
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._eval_collator
        dataloader = super().get_eval_dataloader(eval_dataset)
        self.data_collator = old_collator
        return dataloader

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        old_collator = self.data_collator
        self.data_collator = self._test_collator
        dataloader = super().get_test_dataloader(test_dataset)
        self.data_collator = old_collator
        return dataloader



class TrainerForMMLLM(TrainerDifferentCollatorMixin, Seq2SeqTrainer):
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Override to inject custom behavior.

        # noinspection PyUnresolvedReferences
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # filter keys
        filter_keys = ["labels"]
        for k in inputs:
            if not (k in filter_keys):
                gen_kwargs[k] = inputs[k]
        self._logging_generate_kwargs(gen_kwargs.keys())
        with torch.inference_mode():
            with self.compute_loss_context_manager():
                generated_tokens = self.model.generate(**gen_kwargs)
                # print(type(gen_kwargs))
                # print(generated_tokens)
                # print(generated_tokens[:, inputs['input_ids'].size()[-1]:])
                # print('gen: ', generated_tokens.shape)
                # print('ori: ', gen_kwargs['input_ids'].shape)
                
                # exit()

        # TODO: rewrite official seq2seq_trainer to suppress generation_config warning
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # important for Decoder-Only LLM: only extract generated_tokens and discard origin inputs
        generation_inputs = inputs['input_ids']
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]
        # print('before padding:', generated_tokens)

        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        
        if generated_tokens.shape[-1] < gen_config.max_length:
            # print('max length ', gen_config.max_length)
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            # print('max new length ', gen_config.max_new_tokens)
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)
        
        # print('after padding ', generated_tokens)
        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels
    def _logging_generate_kwargs(self, keys):
        if not hasattr(self, '_generate_kwargs'):
            self._generate_kwargs = None
        if self._generate_kwargs != keys:
            self._generate_kwargs = keys
            logger.warning(f"generate use kwargs: {keys}")

    def save_prediction(self, predict_results, file_key_prefix='predict'):
        if not self.is_world_process_zero():
            return

        import numpy as np
        os.makedirs(self.args.output_dir, exist_ok=True)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_predictions.npy"), predict_results.predictions)
        np.save(os.path.join(self.args.output_dir, f"{file_key_prefix}_label_ids.npy"), predict_results.label_ids)

        preds, targets = predict_results.predictions, predict_results.label_ids
        origin_preds, origin_targets = preds, targets
        preds, targets = deepcopy(preds), deepcopy(targets)
        logger.warning(f"preds shape: {preds.shape}. targets shape: {targets.shape}")

        # decode text and save to json takes forever for big test set
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, f'{file_key_prefix}_extra_prediction.jsonl'), 'a', encoding="utf-8") as g:
            for p, t, pi, ti in tqdm(
                    zip(preds, targets, origin_preds, origin_targets),
                    total=len(preds), desc=f"saving prediction for {file_key_prefix}",
            ):
                p[p < 0] = self.tokenizer.pad_token_id
                t[t < 0] = self.tokenizer.pad_token_id
                p = self.tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                t = self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                obj = dict(
                    pred=p,
                    target=t,
                    # pred_id=pi.tolist(),
                    # target_id=ti.tolist(),
                )
                g.write(json.dumps(obj) + '\n')
                g.flush()

    # transformers + FSDP + saving model -> cuda OOM for small memory gpu
    # refer: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if self.fsdp is not None:
            if output_dir is None:
                output_dir = self.args.output_dir
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                FullStateDictConfig,
                StateDictType,
            )
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state_dict = self.model.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=cpu_state_dict)  # noqa
            # Push to the Hub when `save_model` is called by the user.
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        else:
            super().save_model(output_dir, _internal_call)