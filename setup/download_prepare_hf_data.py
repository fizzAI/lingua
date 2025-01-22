# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16, # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=32):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
        start_method="spawn",
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def main(dataset, memory, data_dir, seed=42, nchunks=32, skipdl=False, skipchunking=False):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
        "minipile": "JeanKaddour/minipile"
    }[dataset]
    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file
    prefix = f"{dataset}.chunk."
    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
        "minipile": ".jsonl"
    }[dataset]
    cat_command = {
        "fineweb_edu": "cat",
        "fineweb_edu_10bt": "cat",
        "dclm_baseline_1.0": "zstdcat",
        "dclm_baseline_1.0_10prct": "zstdcat",
        "minipile": "cat"
    }[dataset]
    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
        "minipile": None
    }[dataset]
    suffix = ".jsonl"
    k_validation = 10000  # Number of lines to take from each chunk for validation

    if not skipdl:
        # Download dataset
        download_dataset(repo_id, src_dir, allow_patterns)

        if "fineweb" in dataset or "minipile" in dataset:
            parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    # Set up environment variables
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # Run the shuffling and splitting using Python since Windows doesn't have Unix tools
    import glob
    import random
    from pathlib import Path
    
    # Get all files matching pattern
    files = glob.glob(str(Path(src_dir) / f'*{orig_extension}'))
    
    if not skipchunking:
        # Read and shuffle all lines
        all_lines = []
        for file in tqdm(files, desc="Reading files"):
            with open(file, 'r', encoding='utf-8') as f:
                all_lines.extend(f.readlines())
        
        random.shuffle(all_lines)
        
        # Split into chunks
        chunk_size = len(all_lines) // nchunks
        chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
        
        # Write chunks to files
        for i, chunk in enumerate(tqdm(chunks, desc="Writing chunks")):
            out_file = Path(out_dir) / f'{prefix}{i:02d}{suffix}'
            with open(out_file, 'w', encoding='utf-8') as f:
                f.writelines(chunk)

    # Create validation set and remove lines from chunks
    validation_file = f"{out_dir}/{dataset}.val{suffix}"
    for i in tqdm(range(nchunks), desc="Creating validation set"):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_validation} {chunk_file} > {validation_file}")
        # Read all lines after k_validation
        run_command(f"tail -n +{k_validation + 1} {chunk_file} > {chunk_file}.tmp && mv {chunk_file}.tmp {chunk_file}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)
    parser.add_argument("--skipdl", action="store_true")
    parser.add_argument("--skipchunking", action="store_true")

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed, args.nchunks, args.skipdl, args.skipchunking)
