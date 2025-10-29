cd ..
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision=bf16 --num_processes=1 \
--num_machines 1 --dynamo_backend no \
-m polygene.main \
--pretrained_model_path 'polygene/model/polygene_architecture.json' \
--eval_data_paths 'data/test_cxg_disease_balanced.h5ad' \
--shard_size 10000 \
--max_length 2007 \
--num_top_genes 58604 \
--vocab_path 'polygene/data_utils/vocab/cxg_phenotypic_tokens_map.json' \
--obs_included_phenotypes disease tissue cell_type sex development_stage assay \
--per_device_eval_batch_size 24 \
--dataloader_num_workers 8 \
--output_dir '/media/lleger/LaCie/mit/runs/polygene_compound' \
--sparse \
--monitor_collapse \
--compound_loss \
--seed 3 \
--use_flash_attn \
mlm \
--gene_mask_prob 0.25 \
--phenotype_mask_prob 0.75 \
--train_data_paths '/media/rohola/ssd_storage/primary/cxg_chunk{1..2501}.h5ad' \
--per_device_train_batch_size 24 \
--learning_rate 1e-4 \
--weight_decay 5e-2 \
--warmup_ratio 0.05 \
--num_train_epochs 2 \
--eval_steps 1 \
--save_steps 50000

