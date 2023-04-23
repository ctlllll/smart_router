from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import transformers
import torch
import numpy as np
from tqdm import tqdm, trange
import argparse
import os
import wandb
# # install open assistant model_training module (e.g. run `pip install -e .` in `model/` directory of open-assistant repository)
# import model_training.models.reward_model

# We instead make a copy of the reward model code here, so that we can import it
import reward_model

class RouterDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenized_dataset = {}

    def __getitem__(self, index):
        if index not in self.tokenized_dataset:
            self.tokenized_dataset[index] = {
                "label": torch.tensor(self.dataset[index]["preference"]),
            }
            tokenized = tokenizer(
                self.dataset[index]["input"],
                return_tensors="pt",
            )
            for key in tokenized:
                self.tokenized_dataset[index][key] = tokenized[key][0]
        return self.tokenized_dataset[index]

    def __len__(self):
        return len(self.dataset)
    

def loss_fct(logits, labels):
    scores = torch.sigmoid(logits)
    loss = torch.nn.functional.l1_loss(scores, labels)
    return loss

class RouterTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fct(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fct(logits.view(-1), labels.view(-1))
        return loss, logits, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5")
    parser.add_argument("--project_name", type=str, default="router_v2")
    parser.add_argument("--run_name", type=str, default="router")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./router")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    model.eval()
    model.cuda()

    router_dataset = torch.load("router_dataset.pt")
    router_dataset = RouterDataset(router_dataset)

    train_dataset, eval_dataset = torch.utils.data.random_split(
        router_dataset,
        [int(len(router_dataset) * 0.9), len(router_dataset) - int(len(router_dataset) * 0.9)],
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(eval_dataset, os.path.join(args.output_dir, "eval_dataset.pt"))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        weight_decay=args.weight_decay,
        report_to="wandb",
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        logging_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        bf16=True,
        save_total_limit=3,
    )

    trainer = RouterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()