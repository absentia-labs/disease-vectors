import os
import itertools
import subprocess
import concurrent.futures

def build_command(seed, n_layers, dim, output_dir_base):
    run_name = f"seed_{seed}_layers_{n_layers}_dim_{dim}"
    output_dir = f"{output_dir_base}_{run_name}"

    command = [
        "accelerate", "launch",
        "--mixed_precision=bf16",
        "--num_processes=1",
        "--num_machines", "1",
        "--dynamo_backend", "no",
        "-m", "polygene.main",

        "--pretrained_model_path", "polygene/model/polygene_architecture.json",
        "--eval_data_paths", "data/test_cxg.h5ad",
        "--shard_size", "10000",
        "--max_length", "2007",
        "--num_top_genes", "58604",
        "--vocab_path", "polygene/data_utils/vocab/cxg_phenotypic_tokens_map.json",
        "--obs_included_phenotypes", "disease", "tissue", "cell_type", "sex", "development_stage", "assay",
        "--per_device_eval_batch_size", "32",
        "--dataloader_num_workers", "8",
        "--output_dir", output_dir,
        "--sparse",
        "--seed", str(seed),
        "--tied",
        "--unit_sphere_constraint",
        "--n_layers", str(n_layers),
        "--dim", str(dim),
        "--use_flash_attn",
        "mlm",
        "--gene_mask_prob", "0.5",
        "--phenotype_mask_prob", "0.75",
        "--train_data_paths", "/media/rohola/ssd_storage/primary/cxg_chunk{0..2502}.h5ad",
        "--per_device_train_batch_size", "32",
        "--learning_rate", "1e-4",
        "--weight_decay", "5e-2",
        "--warmup_ratio", "0.05",
        "--num_train_epochs", "1",
        "--eval_steps", "100000",
        "--save_steps", "100000",
    ]
    return command, run_name

def run_one(task, gpu_id, output_dir_base, repo_cwd):
    seed, n_layers, dim = task
    command, run_name = build_command(seed, n_layers, dim, output_dir_base)

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.run(command, cwd=repo_cwd, env=env)
    return run_name, proc.returncode

if __name__ == "__main__":
    seeds = [2, 3]
    layers = [1, 3, 6]
    dimensions = [96, 144, 240]

    tasks = list({
        (seed, layers[-1], dimensions[-1]) for seed in seeds[:-1]
    } | {
        (seeds[-1], layer, dimensions[-1]) for layer in layers[:-1]
    } | {
        (seeds[-1], layers[-1], dimension) for dimension in dimensions[:-1]
    })
    print(tasks)
    gpu_id = 1
    max_workers = 2

    output_dir_base = "/media/lleger/LaCie/mit/runs/plato/"
    repo_cwd = "../../../"
    failures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, task in enumerate(tasks):
            futures.append(executor.submit(run_one, task, gpu_id, output_dir_base, repo_cwd))

        for future in concurrent.futures.as_completed(futures): # n
            run_name, rc = future.result()


