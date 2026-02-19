"""
train or resume training of a BERT-style masked language model
"""
import os
import pickle

import wandb
import accelerate
import numpy as np
import transformers

from polygene.configs import parse_args, TrainConfig
from polygene.data_utils import (
    DataCollatorForPhenotypicMLM, GeneTokenizer, IterableAnnDataset
)
from polygene.eval.metrics import set_seed
from polygene.model.model import Polygene
from polygene.data_utils.sharded_trainer import ShardedTrainer, UnitSphereConstraint


if __name__ == "__main__":
    config = parse_args()
    set_seed(config.seed)
    distributed_state = accelerate.PartialState()

    tokenizer = GeneTokenizer(config)
    config: TrainConfig = config
    np.random.shuffle(config.train_data_paths)
    
    os.environ.update({ "WANDB_PROJECT": "Disease Geometry", "WANDB_LOG_MODEL": "false", })
    print(config.output_dir)
    working_dir = config.output_dir 
    os.makedirs(working_dir, exist_ok=True) 
    with open(os.path.join(working_dir, "tokenizer.pkl"), "wb") as f: pickle.dump(tokenizer, f)

    # Divide `config.train_data_paths` across the processes. Processes is different from workers handled by accelerate and for multi-GPU
    assert len(config.train_data_paths) % distributed_state.num_processes == 0, "num train paths is multiple of processes"
    shards_per_process = len(config.train_data_paths) // distributed_state.num_processes
    rank = distributed_state.process_index
    process_train_paths = config.train_data_paths[shards_per_process * rank: shards_per_process * (rank + 1)]

    # Load the configuration for a model
    if config.pretrained_model_path.lower().endswith('json'):
        model_config = transformers.AutoConfig.from_pretrained(config.pretrained_model_path)
        # vocab_size and type_vocab_size determine model embeddings row length
        model_config.vocab_size = tokenizer.vocab_size; model_config.type_vocab_size = tokenizer.type_vocab_size
        model_config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        model_config.n_layers = config.n_layers
        model_config.n_heads = 6
        model_config.dim = config.dim
        model_config.hidden_dim = 2*config.dim
        model_config.tied = config.tied
        model_config.unit_sphere_constraint = config.unit_sphere_constraint
        model_kwargs = {"attn_implementation": "flash_attention_2"} if config.use_flash_attn else dict()
        model = Polygene._from_config(model_config, **model_kwargs)

        def count_model_parameters(model):
            print(f"{sum(parameter.numel() for name, parameter in model.named_parameters() if not 'embedding' in name) / 1_000_000:.2f}M")
            print(f"{sum(parameter.numel() for name, parameter in model.named_parameters()) / 1_000_000:.2f}M")
        count_model_parameters(model)

    elif "checkpoint" in config.pretrained_model_path:
        with open(config.pretrained_model_path + "../tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        tokenizer.flexible = True
        tokenizer.neural_updates = {"token_values":[], "token_types": [], "token_value_str":[], "token_type_of_values":[]}
        tokenizer.add_phenotype(config.obs_included_phenotypes)
        model = Polygene.from_pretrained(config.pretrained_model_path,  attn_implementation="flash_attention_2") #torch_dtype=torch.bfloat16

    data_collator = DataCollatorForPhenotypicMLM(
        tokenizer=tokenizer,
        phenotype_mask_prob=config.phenotype_mask_prob,
        genotype_mask_prob=config.gene_mask_prob,
        )
    process_train_paths = np.random.permutation(process_train_paths).tolist()
    train_dataset = IterableAnnDataset(process_train_paths, config, tokenizer)
    eval_dataset = IterableAnnDataset(config.eval_data_paths, config, tokenizer)

    training_args = transformers.TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=True,  # saving tokenizer first
        
        logging_steps=200,
        logging_dir=working_dir,
        logging_first_step=True,

        save_total_limit=config.num_saves,
        save_strategy="steps",
        save_steps=config.save_steps,
        run_name=working_dir.split('/')[-1],
        report_to=["wandb"],
        load_best_model_at_end=False,

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
        
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        metric_for_best_model=config.best_metric,   
        eval_delay=0,
        include_inputs_for_metrics=True,
    )

    trainer = ShardedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[UnitSphereConstraint()] if config.unit_sphere_constraint else None 
    )

    # Train the model
    trainer.train()

    tokenizer.update_from_model_memory(model.config.updates_memory)
    with open(os.path.join(working_dir, "tokenizer.pkl"), "wb") as f: # save last version of tokenizer in case vocab grows
        pickle.dump(tokenizer, f)

    # Evaluate the model
    eval_dataset.tokenizer = tokenizer
    trainer.evaluate()

    wandb.finish()
