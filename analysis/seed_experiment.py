import os
import concurrent.futures
import subprocess

def run_seed(seed):
    base_command = [
        "accelerate", "launch",
        "--mixed_precision=bf16",
        "--num_processes=1",
        "--num_machines", "1",
        "--dynamo_backend", "no",
        "-m", "polygene.main",
        "--pretrained_model_path", "polygene/model/polygene_architecture_small.json",
        "--eval_data_paths", "data/test_cxg.h5ad",
        "--shard_size", "10000",
        "--max_length", "5007",
        "--num_top_genes", "58604",
        "--vocab_path", "polygene/data_utils/vocab/cxg_phenotypic_tokens_map.json",
        "--obs_included_phenotypes", "disease", "tissue", "cell_type", "sex", "development_stage", "assay",
        "--per_device_eval_batch_size", "24",
        "--dataloader_num_workers", "6",
        "--output_dir", f"/media/lleger/LaCie/seeds_experiment/polygene_{seed}",
        "--classification_token",
        "--sparse",
        "--use_flash_attn",
        "--seed", str(seed),
        "mlm",
        "--gene_mask_prob", "0.25",
        "--phenotype_mask_prob", "0.75",
        "--train_data_paths", "/media/rohola/ssd_storage/primary/cxg_chunk{1..1000}.h5ad",
        "--per_device_train_batch_size", "32",
        "--learning_rate", "1e-4",
        "--weight_decay", "5e-2",
        "--warmup_ratio", "0.05",
        "--num_train_epochs", "1",
        "--eval_steps", "100000",
        "--save_steps", "100000"
    ]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    return subprocess.run(base_command, cwd="..", env=env).returncode

if __name__ == "__main__":
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_seed, seed) for seed in range(2)]
            for future in concurrent.futures.as_completed(futures):
                rc = future.result()
                if rc != 0:
                    print(f"Process failed with exit code {rc}")
    except KeyboardInterrupt:
        print("Interrupted! Some jobs may still be running.")
