"""Push trained model to Hugging Face Hub."""

import torch
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import yaml

def push_to_huggingface(
    checkpoint_path: str,
    repo_name: str,
    commit_message: str = "Upload model",
    config_path: str = None
):
    """
    Upload model checkpoint and config to Hugging Face.
    
    Usage:
        python scripts/push_to_hf.py \
            --checkpoint logs/.../checkpoints/last.ckpt \
            --repo username/spatial-mamba-unet \
            --config logs/.../config.yaml
    """
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path)
    
    # Save model weights
    model_path = "model.pth"
    torch.save(ckpt['state_dict'], model_path)
    print(f"Saved model weights to {model_path}")
    
    # Create repo
    print(f"Creating repo: {repo_name}")
    create_repo(repo_name, exist_ok=True)
    
    # Upload files
    api = HfApi()
    
    # Upload model
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pth",
        repo_id=repo_name,
        commit_message=commit_message
    )
    
    # Upload config if provided
    if config_path and Path(config_path).exists():
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.yaml",
            repo_id=repo_name,
            commit_message="Upload config"
        )
    
    print(f"✅ Successfully uploaded to https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--message", type=str, default="Upload model")
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    
    push_to_huggingface(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo,
        commit_message=args.message,
        config_path=args.config
    )