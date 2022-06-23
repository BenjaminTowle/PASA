import torch
import random
import numpy as np
import os

from dataclasses import dataclass, field
from transformers import HfArgumentParser

from engine import get_engine
from dataset import get_dataset
from models import get_model

os.environ["WANDB_DISABLED"] = "true"


@dataclass
class Args:
    # General parameters
    model_type: str = field(default="latent", metadata={"choices": ["baseline", "latent", "distillation"]})
    task_type: str = field(default="clubfloyd", metadata={"choices": ["clubfloyd", "jericho"]})

    # Dataset parameters
    data_directory: str = field(default="calm/cleaned_corpora", metadata={"help": "path to ClubFloyd transcripts."})
    dataset_load_path: str = None
    max_state_length: int = 128
    max_action_length: int = 8
    test_size = 0.1
    num_negatives = 9

    # Model parameters
    negatives: str = field(default="valid", 
        metadata={"help": "Types of negatives to use in contrastive training.", "choices": ["valid", "batch_valid", "batch"]})
    num_codes: int = 5
    intent_type: str = field(default="regex", metadata={"choices": ["latent", "regex", "persona"]})

    # Training parameters
    model_path: str = "distilbert-base-uncased" #"facebook/blenderbot_small-90M" #"huawei-noah/TinyBERT_General_4L_312D"# 
    output_dir: str = "dummy"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    epochs: int = 3
    bsz: int = 8
    seed: int = 27
    device: str = field(default="cuda", metadata={"choices": ["cuda", "cpu"]})
    student_path: str = "clubfloyd_baseline" #"distilbert-base-uncased"
    teacher_path: str = "jericho_regex"
    

def set_random_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

def main():
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    
    set_random_seed(args)

    dataset_dict = get_dataset(args)
    model = get_model(args)

    engine = get_engine(args, model, dataset_dict["train"], dataset_dict["test"])
    engine.train()


if __name__ == "__main__":
    main()
