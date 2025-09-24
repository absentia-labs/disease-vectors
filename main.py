"""
Supports training (either MLM or classification). Inference not implemented yet.
Training works across multiple GPUs on a single machine.
"""
import os
import json
import pickle

import accelerate
import torch
import numpy as np
import torch.nn as nn
import transformers
import wandb

from polygene.configs import parse_args, TrainConfig
from polygene.data_utils import (
    DataCollatorForPhenotypicMLM, GeneTokenizer, IterableAnnDataset
)
from polygene.eval.metrics import preprocess_logits_argmax, metrics_wrapper, set_seed
from polygene.model.model import Polygene
from polygene.data_utils.sharded_trainer import ShardedTrainer


if __name__ == "__main__":
    config = parse_args()
    set_seed(config.seed)
    distributed_state = accelerate.PartialState()

    tokenizer = GeneTokenizer(config)
    config: TrainConfig = config
    np.random.shuffle(config.train_data_paths)
    

    os.environ.update({  # https://docs.wandb.ai/guides/track/environment-variables
        "WANDB_PROJECT": "Polygene",
        "WANDB_LOG_MODEL": "false",
    })

    working_dir = config.output_dir
    os.makedirs(working_dir, exist_ok=True) 

    # Divide `config.train_data_paths` across the processes. Processes is different from workers handled by accelerate and for multi-GPU
    assert len(config.train_data_paths) % distributed_state.num_processes == 0, "num train paths is multiple of processes"
    shards_per_process = len(config.train_data_paths) // distributed_state.num_processes
    rank = distributed_state.process_index
    process_train_paths = config.train_data_paths[shards_per_process * rank: shards_per_process * (rank + 1)]

    # Load the configuration for a model
    if config.pretrained_model_path.lower().endswith('json'):
        model_config = transformers.AutoConfig.from_pretrained(config.pretrained_model_path)
        model_config.vocab_size = tokenizer.vocab_size # vocab_size and type_vocab_size determine model embeddings row length
        model_config.type_vocab_size = tokenizer.type_vocab_size
        model_config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        if config.classification_token: 
            model_config.obs_included_phenotypes = tokenizer.phenotypic_types
            model_config.phenotypic_tokens_map = tokenizer.phenotypic_tokens_map
            model_config.n_bins = tokenizer.num_bins + 1
        model_config.classification_token = config.classification_token
        model_kwargs = {"attn_implementation": "flash_attention_2"} if config.use_flash_attn else dict()
        model = Polygene._from_config(model_config, **model_kwargs)
    elif "checkpoint" in config.pretrained_model_path:
        with open(config.pretrained_model_path + "../tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        tokenizer.flexible=True
        tokenizer.neural_updates = {"token_values":[], "token_types": [], "token_value_str":[], "token_type_of_values":[]}
        tokenizer.add_phenotype(config.obs_included_phenotypes)
        model = Polygene.from_pretrained(config.pretrained_model_path, attn_implementation="flash_attention_2")

    data_collator = DataCollatorForPhenotypicMLM(
        tokenizer=tokenizer,
        phenotype_mask_prob=config.phenotype_mask_prob,
        genotype_mask_prob=config.gene_mask_prob,
        )
    process_train_paths = np.random.permutation(process_train_paths).tolist() # not true shuffling but minimizes successive shard variance
    train_dataset = IterableAnnDataset(process_train_paths, config, tokenizer)

    eval_dataset = IterableAnnDataset(config.eval_data_paths, config, tokenizer)
    eval_metric = metrics_wrapper(model, tokenizer)

    # TODO: shuffle train, add shard_sizes arg, diff shard_size for eval

    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {model_total_params}")

    training_args = transformers.TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=True,  # saving tokenizer first
        
        logging_steps=config.per_device_train_batch_size * 4,
        logging_dir=working_dir,
        logging_first_step=True,

        save_total_limit=config.num_saves,
        save_strategy="steps",
        save_steps=config.save_steps,
        run_name=working_dir.split('/')[-1],
        report_to=["wandb"],
        load_best_model_at_end=True,

        # training stability parameters
        warmup_ratio=config.warmup_ratio,
        learning_rate=config.learning_rate,
        max_grad_norm=1,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 0.1*config.learning_rate}, 
        weight_decay=config.weight_decay,

        num_train_epochs=config.num_train_epochs,

        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        dataloader_num_workers=config.dataloader_num_workers,
        accelerator_config={"dispatch_batches": False},
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        metric_for_best_model=config.best_metric,   
    )

    trainer = ShardedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=eval_metric,
        preprocess_logits_for_metrics=preprocess_logits_argmax,
    )

    # Train the model
    trainer.train()

    tokenizer.update_from_model_memory(model.config.updates_memory)
    with open(os.path.join(working_dir, "tokenizer.pkl"), "wb") as f: # Save last version of tokenizer
        pickle.dump(tokenizer, f)

    # Evaluate the model
    eval_dataset.tokenizer = tokenizer
    trainer.evaluate()

    wandb.finish()
    os._exit(0)
