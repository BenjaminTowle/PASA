import torch
import logging
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from statistics import mean
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Engine:
    def __init__(self, args, model, train_dataset=None, eval_dataset=None, metrics=None) -> None:
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = Adam(model.parameters(), lr=args.learning_rate)
        self.metrics = metrics

    def process_batch(self, batch):
        batch["input_ids"] = torch.stack(batch["input_ids"]).to(self.device).transpose(0, 1)
        batch["attention_mask"] = torch.stack(batch["attention_mask"]).to(self.device).transpose(0, 1)

        batch["act_input_ids"] = torch.stack([torch.stack(p) for p in batch["act_input_ids"]]).to(self.device).transpose(
                    0, 1).transpose(0, -1)
        batch["act_attention_mask"] = torch.stack([torch.stack(p) for p in batch["act_attention_mask"]]).to(self.device).transpose(
                    0, 1).transpose(0, -1)


        return batch

    def train_pass(self, train_dl, eval_dl):
        self.model.train()
        losses = []
        for i, batch in enumerate(tqdm(train_dl)):
            batch = self.process_batch(batch)

            loss = self.model(**batch).loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if (i+1) % 3000 == 0:
                eval_loss, eval_acc = self.eval_pass(eval_dl)
                self.model.train()

                print(f"Eval: loss = {eval_loss}; acc = {eval_acc}")
                
                self.model.save_pretrained(self.args.output_dir) 

                

        total_loss = mean(losses)

        return total_loss

    def eval_pass(self, eval_dl):
        self.model.eval()
        losses = []
        accs = []
        mis = []
        with torch.no_grad():
            for batch in tqdm(eval_dl):
                batch = self.process_batch(batch)

                outputs = self.model(**batch)
                loss = outputs.loss

                pred = [s.argmax().item() for s in outputs.logits]
                acc = accuracy_score(batch["labels"].cpu().numpy(), pred)

                losses.append(loss.item())
                accs.append(acc)

                #if outputs.mi is not None:
                    #mis.append(outputs.mi.item())

        total_loss = mean(losses)
        total_acc = mean(accs)
        #if mis != []:
            #print("Mutual Information: ", mean(mis))

        return total_loss, total_acc


    def train(self):
        """Main train function"""
        train_dl = DataLoader(self.train_dataset, batch_size=self.args.bsz, shuffle=True)
        eval_dl = DataLoader(self.eval_dataset, batch_size=self.args.bsz, shuffle=False)

        for i in range(self.args.epochs):
            train_loss = self.train_pass(train_dl, eval_dl)

            print(f"Epoch {i}: loss = {train_loss}")

            eval_loss, eval_acc = self.eval_pass(eval_dl)

            print(f"Eval: loss = {eval_loss}; acc = {eval_acc}")
                
            self.model.save_pretrained(self.args.output_dir)
  
       

class PretrainingEngine(Engine):
    
    def process_batch(self, batch):
        batch["input_ids"] = torch.stack(batch["input_ids"]).to(self.device).transpose(0, 1)
        batch["act_input_ids"] = torch.stack(batch["act_input_ids"]).to(self.device).transpose(0, 1)
        batch["attention_mask"] = torch.stack(batch["attention_mask"]).to(self.device).transpose(0, 1)
        batch["act_attention_mask"] = torch.stack(batch["act_attention_mask"]).to(self.device).transpose(0, 1)

        if self.args.model_type == "baseline":
            batch.pop("intents")
            batch.pop("persona_ids")

        if not self.model.training:
            batch["cand_input_ids"] = torch.stack([torch.stack(p) for p in batch["cand_input_ids"]]).to(self.device).transpose(
                    0, 1).transpose(0, -1)
            batch["cand_attention_mask"] = torch.stack([torch.stack(p) for p in batch["cand_attention_mask"]]).to(self.device).transpose(
                    0, 1).transpose(0, -1)


        return batch

def get_engine(args, model, train_dataset, eval_dataset):

    if args.task_type == "clubfloyd":
        return PretrainingEngine(args, model, train_dataset, eval_dataset)
    elif args.task_type == "jericho":
        return Engine(args, model, train_dataset, eval_dataset)
    
    raise ValueError("task_type not recognised!")
