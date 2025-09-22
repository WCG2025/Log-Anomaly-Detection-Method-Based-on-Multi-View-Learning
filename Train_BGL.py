import os
import torch
import argparse
from datetime import datetime
import random
import numpy as np
import pandas as pd
import csv
from datasets import load_dataset, Dataset

from transformers import T5TokenizerFast, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm.auto import tqdm

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=40)
parser.add_argument("--model", type=str, default="flan-t5-base")
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--validation", type=str, default="validation")
args = parser.parse_args()

# 设置环境和随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = True
seed = 41
random.seed(seed)
batch_size = args.batch_size
lr = args.learning_rate
num_epochs = args.num_epochs
model_name = "flan-t5-base"
pretrainedmodel_path = "/root/autodl-tmp/Graduation/Flan-T5"

# 数据预处理
def prepare_data():
    # 读取训练集、验证集和测试集
    dataset_path = "/root/autodl-tmp/Graduation/BGL_test_data/test.json"
    raw_dataset = pd.read_json(dataset_path)
    raw_dataset = raw_dataset.drop(columns=['instruction'])
    raw_dataset = raw_dataset.applymap(str)  # 转换为字符串
    new_column_names = {'input': 'Content', 'output': 'EventTemplate'}
    raw_dataset.rename(columns=new_column_names, inplace=True)

    # 同样的处理方式处理验证集和测试集
    validation_path = "/root/autodl-tmp/Graduation/BGL_validation_data/validation.json"
    validation_dataset = pd.read_json(validation_path)
    validation_dataset = validation_dataset.drop(columns=['instruction'])
    validation_dataset = validation_dataset.applymap(str)  # 转换为字符串
    validation_dataset.rename(columns=new_column_names, inplace=True)

    test_path = "/root/autodl-tmp/Graduation/BGL_test_data/test.json"
    test_dataset = pd.read_json(test_path)
    test_dataset = test_dataset.drop(columns=['instruction'])
    test_dataset = test_dataset.applymap(str)  # 转换为字符串
    test_dataset.rename(columns=new_column_names, inplace=True)

    # 转换为 Huggingface Dataset 格式
    train_val_test = {
        "train": Dataset.from_dict(raw_dataset),
        "validation": Dataset.from_dict(validation_dataset),
        "test": Dataset.from_dict(test_dataset)
    }
    return train_val_test

# 创建用于 DataLoader 的 Dataset 类
class LogDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_content= self.data[idx]["Content"]
        content = "Parse the raw log to log template: "+self.data[idx]["Content"]
        event_template = self.data[idx]["EventTemplate"]
        encoding = self.tokenizer(content, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        labels = self.tokenizer(event_template, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")["input_ids"]
        return {
            "raw_content": raw_content,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

# 加载预训练模型和分词器
new_words = ["<*>", "{", "}", "<", "\\"]
tokenizer = T5TokenizerFast.from_pretrained(pretrainedmodel_path)
tokenizer.add_tokens(new_tokens=new_words)
model = T5ForConditionalGeneration.from_pretrained(pretrainedmodel_path)
model.to('cuda')
# 准备数据
dataset = prepare_data()
train_dataset = LogDataset(dataset["train"], tokenizer)
validation_dataset = LogDataset(dataset["validation"], tokenizer)
test_dataset = LogDataset(dataset["test"], tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

# 优化器和学习率调度器设置
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 训练循环
output_dir = "/root/autodl-tmp/Graduation/BGL_Flan_T5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_time=datetime.now()

def acc(string1, string2):
    string1 = string1.replace(" ", "")
    string2 = string2.replace(" ", "")

    if string1 == string2:
        return True
    else:
        return False
diff_out_path = "/root/autodl-tmp/Graduation/output"
progress_bar = tqdm(range(num_training_steps), desc="Training Progress")
best_val_acc = 0
for epoch in range(num_epochs):
    model.train()
    tot_loss = 0
    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to("cuda" if use_cuda else "cpu")
        attention_mask = batch["attention_mask"].to("cuda" if use_cuda else "cpu")
        labels = batch["labels"].to("cuda" if use_cuda else "cpu")

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        tot_loss += loss.item()

        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        progress_bar.update(1)

    print(f"Epoch {epoch}, avergeLoss: {tot_loss / (step + 1)}")

    # 验证阶段
    model.eval()
    if args.validation == "validation" and (epoch + 1) % 5 == 0:
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in validation_dataloader:
                input_ids = batch["input_ids"].to("cuda" if use_cuda else "cpu")
                attention_mask = batch["attention_mask"].to("cuda" if use_cuda else "cpu")
                labels = batch["labels"].to("cuda" if use_cuda else "cpu")

                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=256)
                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                ground_truths = tokenizer.batch_decode(labels, skip_special_tokens=True)

                for prediction, ground_truth in zip(predictions, ground_truths):
                    if acc(prediction,ground_truth)==1:
                        correct_preds += 1
                    total_preds += 1

        accuracy = correct_preds / total_preds

        if accuracy >= best_val_acc:
            predictions = []
            ground_truths = []
            csv_out = []
            res_out = []
            cnt=0
            best_val_acc = accuracy
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to("cuda" if use_cuda else "cpu")
                attention_mask = batch["attention_mask"].to("cuda" if use_cuda else "cpu")
                labels = batch["labels"].to("cuda" if use_cuda else "cpu")

                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=256)
                pres = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                grounds = tokenizer.batch_decode(labels, skip_special_tokens=True)
                inputs_decode =tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                predictions = [pre.strip() for pre in pres]
                ground_truths = [ground.strip() for ground in grounds]
                for i in range(len(inputs_decode)):
                    if not acc(predictions[i], ground_truths[i]):
                        cnt=cnt+1
                        csv_out.append([inputs_decode[i], predictions[i], ground_truths[i]])
                if not os.path.exists(diff_out_path):
                    os.makedirs(diff_out_path)
                for i in range(len(inputs_decode)):
                    res_out.append([inputs_decode[i], predictions[i], ground_truths[i]])
                with open(diff_out_path + "prediction.csv", "w") as f:
                    f = csv.writer(f)
                    f.writerows(res_out)
                with open(diff_out_path + "diff.csv", "w") as f:
                    f = csv.writer(f)
                    f.writerows(csv_out)
                with open(diff_out_path + "result.txt", "w") as f:
                    f.write(str(1-cnt/2000))
            print("acc: {}".format(1-cnt/2000))



# 测试阶段
# if args.validation != "validation":
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)

#     model.eval()
#     correct_preds = 0
#     total_preds = 0
#     with torch.no_grad():
#         for batch in test_dataloader:
#             input_ids = batch["input_ids"].to("cuda" if use_cuda else "cpu")
#             attention_mask = batch["attention_mask"].to("cuda" if use_cuda else "cpu")
#             labels = batch["labels"].to("cuda" if use_cuda else "cpu")

#             outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
#             predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             ground_truths = tokenizer.batch_decode(labels, skip_special_tokens=True)

#             for prediction, ground_truth in zip(predictions, ground_truths):
#                 if prediction == ground_truth:
#                     correct_preds += 1
#                 total_preds += 1

#     test_accuracy = correct_preds / total_preds
#     print(f"Test Accuracy: {test_accuracy}")

finish_time = datetime.now()
duration = finish_time - start_time
print(f"Training complete in {duration}")
