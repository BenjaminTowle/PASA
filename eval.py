import torch
import numpy as np
import pickle

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel
from statistics import mean

from dataset import get_dataset
from train import set_random_seed
from models import get_model

@dataclass
class Args:
    model_path: str = "distillation"
    model_type: str = field(default="baseline", metadata={"choices": ["baseline", "latent", "random"]})
    task_type: str = field(default="jericho", metadata={"choices": ["jericho"]})

    dataset_load_path: str = None

    max_state_length: int = 128
    max_action_length: int = 8

    compare_path: str = None

    seed: int = 0
    bsz: int = 8

    device: str = field(default="cuda", metadata={"choices": ["cuda", "cpu"]})


def process_batch(batch, device="cuda"):
    batch["input_ids"] = torch.stack(batch["input_ids"]).to(device).transpose(0, 1)
    batch["attention_mask"] = torch.stack(batch["attention_mask"]).to(device).transpose(0, 1)

    batch["act_input_ids"] = torch.stack([torch.stack(p) for p in batch["act_input_ids"]]).to(device).transpose(
                    0, 1).transpose(0, -1)
    batch["act_attention_mask"] = torch.stack([torch.stack(p) for p in batch["act_attention_mask"]]).to(device).transpose(
                    0, 1).transpose(0, -1)

    return batch


def get_game_statistics(dataset):
    stats = {}
    stats["avg_valid_acts"] = mean(dataset["act_sizes"])
    stats["num_samples"] = len(dataset)
    stats["num_reward_samples"] = sum([1 for i in range(len(dataset)) if dataset[i]["rew"] > 0 ])

    return stats



from transformers import BertTokenizer

t = BertTokenizer.from_pretrained("bert-base-uncased")

def evaluate_game(model, dataset, batch_size=8):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels = []
    preds = []
    rewards = []
    sizes = []
    ranks = []
    
    for batch in dl:
        batch = process_batch(batch)
        rewards += batch["rew"].cpu().numpy().tolist()
        batch.pop("rew")
        batch.pop("game")
        batch.pop("intents")
        outputs = model(**batch)
        pred = [s.argmax().item() for s in outputs.logits]
        rank = [s.argsort(0, descending=True)[0].item() for s in outputs.logits]

        sizes += batch["act_sizes"].cpu().numpy().tolist()

        labels += batch["labels"].cpu().numpy().tolist()
        preds += pred
        ranks += rank

    return preds, labels, rewards, sizes, ranks
    









def main():
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    dataset_dict = get_dataset(args)
    eval_dataset = dataset_dict["test"]
    model = get_model(args).to(args.device)
    model.eval()

    set_random_seed(args)  # Shouldn't be any randomness, but just in case ...

    torch.set_grad_enabled(False)

    # Get list of games
    games = []
    for i in range(len(eval_dataset)):
        game = eval_dataset[i]["game"]
        if game not in games:
            games.append(game)
    
    # Create separate datasets for each game
    game2dataset = {game:eval_dataset.filter(lambda x: x["game"] == game) for game in games}

    # Collect statistics for each game
    for game, dataset in game2dataset.items():
        stats = get_game_statistics(dataset)
        print(game)
        print(stats)
        print("======================")
    
    all_stats = get_game_statistics(eval_dataset)
    print(all_stats)

    # Collect metrics for each game
    all_preds, all_labels, all_rewards, all_sizes, all_ranks = [], [], [], [], []
    for game, dataset in game2dataset.items():
        preds, labels, rewards, sizes, ranks = evaluate_game(model, dataset, batch_size=args.bsz)

        accs = np.array([1 if p == l else 0 for p, l in zip(preds, labels)])
        acc = np.mean(accs)

        all_preds += preds
        all_labels += labels
        all_sizes += sizes
        all_ranks += ranks

        print(game)
        print("acc: ", acc)
        print("=========================")

    accs = np.array([1 if p == l else 0 for p, l in zip(all_preds, all_labels)])
    acc = np.mean(accs)

    pickle.dump(accs, open(f"{args.model_path}_pred.pkl", "wb"))

    print("acc: ", acc)

    if args.compare_path is not None:
        comp_pred = pickle.load(open(f"{args.compare_path}_pred.pkl", "rb"))
        p_pred = ttest_rel(accs, comp_pred)
        
        print("pred: ", p_pred.pvalue)
    

if __name__ == "__main__":
    main()
