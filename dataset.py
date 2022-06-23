import random
import os

from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer

from intents import get_intents

def split_into_states_actions(lines):
    """
    Splits unprocessed lines into two lists of states and actions, removing the overlap
    """
    states = []
    actions = []
    for line in lines:
        line = line.replace("[STATE]", "[SEP]").replace("[ACTION]", "[SEP]").split("[SEP]")[1:]
        assert len(line) == 4, "line should consist of state, action, state, action"
        state = "[SEP]".join(line[:3])
        action = line[3]
        states.append(state)
        actions.append(action)

    return states, actions


def process_clubfloyd(files, tokenizer, args) -> Dataset:
    dataset_dict = {}
    random.shuffle(files)
    idx = int(len(files)*args.test_size)
    train_files = files[:-idx]
    test_files = files[-idx:]

    for split, fs in [("train", train_files), ("test", test_files)]:

        #print(len(fs))
        #exit()

        all_states = []
        all_actions = []
        persona_ids = []
        for i, file in enumerate(fs):
            with open(os.path.join(args.data_directory, file)) as f:
                lines = f.readlines()
                states, actions = split_into_states_actions(lines)
                all_states += states
                all_actions += actions
                persona_ids += [i] * len(actions)

        intents = get_intents(all_actions)

        with open("actions.txt", "w") as af:
            for i, a in zip(intents, all_actions):
                if i == 3:
                    af.write(a)

        # Add blenderbot special tokens
        if args.model_type == "generator":
            all_actions = [tokenizer.bos_token + a + tokenizer.eos_token for a in all_actions]

        states_inputs = tokenizer(all_states, max_length=args.max_state_length, truncation=True, padding="max_length")
        actions_inputs = tokenizer(all_actions, max_length=args.max_action_length, truncation=True, padding="max_length")

        dict_ = {
            "input_ids": states_inputs.input_ids, 
            "attention_mask": states_inputs.attention_mask,
            "act_input_ids": actions_inputs.input_ids,
            "act_attention_mask": actions_inputs.attention_mask,
            "intents": intents,
            "persona_ids": persona_ids
        }
        
        if split == "test":
            cand_input_ids = []
            cand_attention_mask = []
            for i in range(len(all_actions)):
                input_ids = []
                attention_mask = []
                for j in range(args.num_negatives):
                    idxs = [i]
                    while True:
                        idx = random.randint(0, len(all_actions) - 1)
                        if idx not in idxs:
                            idxs.append(idx)
                            break
                    input_ids.append(actions_inputs.input_ids[idx])
                    attention_mask.append(actions_inputs.attention_mask[idx])
                cand_input_ids.append(input_ids)
                cand_attention_mask.append(attention_mask)
            
            dict_["cand_input_ids"] = cand_input_ids
            dict_["cand_attention_mask"] = cand_attention_mask
            dict_["labels"] = [0 for _ in all_actions]

        dataset = Dataset.from_dict(dict_)

        dataset_dict[split] = dataset

    return dataset_dict


def get_dataset_clubfloyd(args, tokenizer):
    
    # Jericho games we want to exclude from data
    exclude = ['intfic_clubfloyd_20090402.html', \
                    'intfic_clubfloyd_20090904.html', \
                    'intfic_clubfloyd_20160401.html', \
                    'intfic_clubfloyd_20160401.txt', \
                    'intfic_clubfloyd_20160701.html', \
                    'intfic_clubfloyd_20161102.html', \
                    'intfic_clubfloyd_20170104.html', \
                    'intfic_clubfloyd_20100903.html', \
                    'intfic_clubfloyd_20080601.html', \
                    "intfic_clubfloyd_20140103.html"]  # this one is blank
    files = [s for s in os.listdir(args.data_directory) if s not in exclude]

    dataset_dict = process_clubfloyd(files, tokenizer, args)

    dataset_dict = DatasetDict({"train": dataset_dict["train"], "test": dataset_dict["test"]})

    return dataset_dict


def map_fn(samples, args, tokenizer):
    # For test we include each valid action with broadcast observation
    act_sizes = [len(v) for v in samples["valid_act"]]

    intent = get_intents([v[0] for v in samples["valid_act"]])

    assert all([s > 0 for s in act_sizes])

    # PAD actions with dummy actions
    max_actions = max(act_sizes)
    for i, size in enumerate(act_sizes):
        if max_actions > size:
            samples["valid_act"][i] += [tokenizer.pad_token] * (max_actions - size)

    # PAD next_observations
    for i, size in enumerate(act_sizes):
        if max_actions > size:
            samples["next_obs"][i] += [tokenizer.pad_token] * (max_actions - size)

    actions = [a for v in samples["valid_act"] for a in v]

    actions = tokenizer(actions, max_length=args.max_action_length, padding="max_length", truncation=True)

    # Broadcast obs
    # remove [CLS] but keep [SEP]
    observations = tokenizer(samples["obs"], max_length=args.max_state_length, padding="max_length", truncation=True)

    labels = [0 for _, _ in zip(samples["act"], samples["valid_act"])]

    # Refold
    def refold(ids):
        new_ids = []
        for size in act_sizes:
            new_ids.append(ids[:max_actions])
            del ids[:max_actions]
        return new_ids

    action_input_ids = refold(actions.input_ids)
    action_attn_mask = refold(actions.attention_mask)
    
    samples["input_ids"] = observations.input_ids
    samples["attention_mask"] = observations.attention_mask
    samples["act_input_ids"] = action_input_ids
    samples["act_attention_mask"] = action_attn_mask
    samples["labels"] = labels
    samples["act_sizes"] = act_sizes
    samples["intents"] = intent

    return samples


def get_dataset_jericho(args, tokenizer):

    dataset_dict = {}
    for split, path in [("train", "train_game.jsonl"), ("test", "test_game.jsonl")]:
        dataset = Dataset.from_json(path)
        rmv_cols = ["obs", "next_obs", "act", "valid_act"]

        dataset = dataset.map(
            lambda x: map_fn(x, args, tokenizer), batched=True, batch_size=len(dataset), remove_columns=rmv_cols, load_from_cache_file=False)

        dataset_dict[split] = dataset

    dataset_dict = DatasetDict({"train": dataset_dict["train"], "test": dataset_dict["test"]})

    return dataset_dict

from transformers import BlenderbotSmallTokenizer

def get_dataset(args):
    
    if args.dataset_load_path is not None:
        return DatasetDict.load_from_disk(args.dataset_load_path)

    if args.model_type == "generator":
        tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
    else:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    if args.task_type == "clubfloyd":
        return get_dataset_clubfloyd(args, tokenizer)
    elif args.task_type == "jericho":
        return get_dataset_jericho(args, tokenizer)
    
    raise ValueError("task_type not recognised!")
