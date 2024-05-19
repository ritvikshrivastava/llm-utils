# Install relevant libraries: pip install transformers datasets evaluate accelerate
# https://jaotheboss.medium.com/peft-with-bert-8763d8b8a4ca


from typing import Any

import numpy as np
import evaluate
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


class BERTFineTuner:
    def __init__(self, bert_model_variant: str = None):
        self.lora_model = None
        self.model_name = bert_model_variant if bert_model_variant else 'bert-base-uncased'

    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @property
    def model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name)

    @property
    def lora_config(self):
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,  # this is necessary
            inference_mode=True
        )

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

    @staticmethod
    def compute_metrics(eval_pred, metric: str = "accuracy") -> dict[str, Any]:
        metric = evaluate.load(metric)
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def finetune(self, train_dataset, eval_dataset):
        tokenized_train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)

        # add LoRA adaptor
        self.lora_model = get_peft_model(self.model, self.lora_config)
        self.lora_model.print_trainable_parameters()

        training_args = TrainingArguments(output_dir="bert_lora_trainer")
        bert_lora_trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,  # training dataset requires column input_ids
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        bert_lora_trainer.train()
        bert_lora_trainer.model.save_pretrained("bert_lora")

    def merge_model(self):
        adapted_model = PeftModel.from_pretrained(
            self.model, "bert_lora"  # bert-lora; the folder of the saved adapter
        )
        merged_model = adapted_model.merge_and_unload()
        merged_model.save_pretrained("merged-bert-lora-model")


def get_yelp_data_sample(split: str = "train", sample: int = 1000) -> Dataset:
    dataset = load_dataset("yelp_review_full")
    if split == "train":
        return dataset["train"].shuffle(seed=42).select(range(sample))
    return dataset["test"].shuffle(seed=42).select(range(sample))


def tune_bert():
    yelp_train_dataset = get_yelp_data_sample("train")
    yelp_test_dataset = get_yelp_data_sample("test")
    bft = BERTFineTuner()
    bft.finetune(train_dataset=yelp_train_dataset, eval_dataset=yelp_test_dataset)


if __name__ == "__main__":
    tune_bert()
