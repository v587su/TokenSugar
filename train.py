import os
import random
from dataclasses import dataclass, field
from typing import cast
from huggingface_hub import login
import copy

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

token = os.getenv('HUGGINGFACE_TOKEN')
if token is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
login(token=token)


import torch
from sugar_dataset import BaseDataset
from transformers import HfArgumentParser, Trainer, TrainingArguments
from datasets import concatenate_datasets
from llm_wrapper import (
    DecodingConfig,
    EncodingConfig,
    TokenizationContext,
    get_model_context,
    pad_sequences,
)



count = os.cpu_count()
N_CORES = 1 if count is None or count == 0 else count // 2

@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100


def map_dataset(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    instructions = examples["instruction"]
    responses = examples["response"]
    prompts = [
            MAGICODER_PROMPT.format(instruction=instruction, response="")
        for instruction in instructions
    ]
    completions = responses

    assert len(prompts) == len(completions)
    prompt_config = EncodingConfig(add_bos=True, add_eos=False)
    completion_config = EncodingConfig(add_bos=False, add_eos=True)
    prompt_id_batches = context.encode(prompt_config, prompts)
    completion_id_batches = context.encode(completion_config, completions)
    assert len(prompt_id_batches) == len(completion_id_batches)
    untruncated_input_ids = [
        (instruction_ids + response_ids)
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    exceeding_length = [
        len(input_id) > args.max_training_seq_length
        for input_id in untruncated_input_ids
    ]
    input_ids = [
        input_id[: args.max_training_seq_length] for input_id in untruncated_input_ids
    ]
    # NOTE: no need to set EOF to IGNORED_INDEX as it is *implicitly* ignored inside
    # the model.forward that shifts the logits left by 1
    labels = [
        (list(map(lambda _: IGNORED_INDEX, instruction_ids)) + response_ids)[
            : args.max_training_seq_length
        ]
        for instruction_ids, response_ids in zip(
            prompt_id_batches, completion_id_batches
        )
    ]
    # `len` of each returned value must be the same, which is required by `tokenizer.map`
    # After `map`, they are treated as individual pieces of data, not as a batch.
    assert len(input_ids) == len(labels)
    for input_id_batch, label_batch in zip(input_ids, labels):
        assert len(input_id_batch) == len(label_batch)
    # print(context.decode(DecodingConfig.default(), input_ids[0:])[0])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "exceeding_length": exceeding_length,
    }

def map_pretrain_dataset(examples: dict[str, list[str]], args: "Args", context: TokenizationContext):
    config = EncodingConfig(add_bos=False, add_eos=False)
    tokenized_example = context.encode(config, examples['content'])
    concatenated_examples = {}
    all_token_ids = []
    for tokenized_input in tokenized_example:
        all_token_ids.extend(tokenized_input + [context.eos_token_id])
    concatenated_examples['input_ids'] = all_token_ids
        
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    result = {k:[] for k in concatenated_examples.keys()}
    for k,t in concatenated_examples.items():
        for i in range(0, total_length, args.max_training_seq_length):
            if i+args.max_training_seq_length < total_length:
                result[k].append(t[i:i+args.max_training_seq_length])
    return {'input_ids': result['input_ids'], 'labels': result["input_ids"].copy()}


def get_data_collator(args: "Args", pad_token_id: int):
    """Pad input_ids to the right, create labels by setting the padding tokens to -100, and
    create attention_mask to ignore the padding tokens"""

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        padding_length = (
            args.max_training_seq_length if args.pad_to_max_length else None
        )
        input_ids = pad_sequences(
            input_ids_unpadded, pad_token_id, "right", padding_length=padding_length, 
        )
        labels = pad_sequences(
            labels_unpadded, IGNORED_INDEX, "right", padding_length=padding_length
        )

        assert input_ids.shape == labels.shape
        assert len(input_ids) == len(examples)
        # Enforced in `map_raw_dataset`
        assert input_ids.shape[-1] <= args.max_training_seq_length
        if args.pad_to_max_length:
            assert input_ids.shape[-1] == args.max_training_seq_length

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
        }
    return collate


@dataclass
class Args:
    datafile_paths: list[str] = field(default_factory=list)
    pattern_file_path: str = field(default=None, metadata={"help": "path to the pattern file"})
    max_training_seq_length: int = field(default=1216)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )
    use_flash_attention: bool = field(default=False)
    is_sugarized: bool = field(default=True)
    load_from: str = field(default=None, metadata={"help": "disk, hub, or None"})
    num_proc: int = field(default=N_CORES)
    task: str = field(default=None, metadata={"help": "name of the task"})
    source_data: str = field(default='magicoder', metadata={"help": "source data to load"})
    use_lora: bool = field(default=False, metadata={"help": "use lora"})
    start_id: int = field(default=None, metadata={"help": "start id"})
    joint_ratio: float = field(default=1.0, metadata={"help": "joint ratio"})

def train():

    parser = HfArgumentParser((ModelArguments, TrainingArguments, Args))
    model_args, training_args, args = cast(
        tuple[ModelArguments, TrainingArguments, Args],
        parser.parse_args_into_dataclasses(),
    )
    print(f'args: {args}')
    print(f'training args: {training_args}')
    if 'sugarized' in model_args.model_key:
        assert args.start_id is not None, "start id is required for sugarized model"
 
    if args.task == "pretrain":
        data_wrapper = BaseDataset(args)
        dataset = data_wrapper.dataset
    elif args.task == "joint_pretrain":
        data_wrapper = BaseDataset(args)
        fake_args = copy.deepcopy(args)
        fake_args.is_sugarized = False
        fake_args.load_from = 'new'
        original_dataset = BaseDataset(fake_args).dataset
        if args.joint_ratio <= 1.0:
            full_index = list(range(len(original_dataset)))
            desugarized_index = random.sample(full_index, int(len(original_dataset) * args.joint_ratio))
            desugarized_dataset = original_dataset.select(desugarized_index)
            sugarized_dataset = data_wrapper.dataset.select(list(set(full_index) - set(desugarized_index)))
            dataset = concatenate_datasets([sugarized_dataset, desugarized_dataset])
        dataset = dataset.shuffle(seed=training_args.seed)
    else:
        raise ValueError(f"Invalid task: {args.task}")
    print(f'num special tokens: {len(data_wrapper.special_tokens)}')


    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key

    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    if args.is_sugarized:
        tokenization_context.set_special_tokens(data_wrapper.special_tokens, start_id=args.start_id)

        print(f'special tokens map: {tokenization_context.special_token_map}')
    save_path = os.path.join(training_args.output_dir, "best_model")
    tokenization_context.tokenizer.save_pretrained(save_path)

    if args.count_tokens:
        print(f'total tokens: {data_wrapper.get_token_num(tokenization_context.tokenizer)}')
        return
    train_dataset = dataset.map(
        function=map_dataset if "pretrain" not in args.task else map_pretrain_dataset,
        fn_kwargs=dict(args=args, context=tokenization_context),
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,  # not args.overwrite_cache
        desc="Running tokenizer on train dataset",
    )


    # Shuffling
    if training_args.eval_steps is None and training_args.evaluation_strategy == "no":
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        eval_dataset = None
    else:
        print("Splitting dataset")
        split_dataset = train_dataset.train_test_split(
            test_size=args.eval_dataset_size,
            shuffle=True,
            seed=training_args.seed,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    state = get_model_context(
        model_key,
        model_name_or_path,
        tokenization_context,
        inference_mode=False,
        use_flash_attention=args.use_flash_attention,
    )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        state.model = get_peft_model(state.model, lora_config)
        

    if args.freeze_non_embeddings:
        for param in state.model.parameters():
            param.requires_grad = False
        for param in state.model.model.embed_tokens.parameters():
            param.requires_grad = True
        for param in state.model.lm_head.parameters():
            param.requires_grad = True
    elif args.freeze_non_head_and_tails:
        # freeze all layers execpt of the first three layers and the last three layers
        # gpt_neox 
        for param in state.model.parameters():
            param.requires_grad = True
        for param in state.model.gpt_neox.layers[args.frozen_layers:-args.frozen_layers]:
            param.requires_grad = False
    elif args.freeze_non_tails:
        for param in state.model.parameters():
            param.requires_grad = True
        for param in state.model.model.layers[:-args.frozen_layers]:
            param.requires_grad = False

    print("Parallel mode:", training_args.parallel_mode)
    data_collator = get_data_collator(args, state.tokenization_context.pad_token_id)

    # neftune_noise_alpha
    trainer = Trainer(
        model=state.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )
    # NOTE: the checkpoint will override the initialized model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    save_path = os.path.join(training_args.output_dir, "best_model")
    trainer.save_model(save_path)
    state.tokenization_context.tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    train()