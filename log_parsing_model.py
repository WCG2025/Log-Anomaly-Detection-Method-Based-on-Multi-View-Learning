import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class TrainData(Dataset):
    def __init__(self, label_content_tuples, tokenizer, max_length=512):
        self.data = label_content_tuples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, content, eventtemplate = self.data[idx]
        
        # 编码输入内容
        input_text = content
        encoding = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        # 编码标签
        label_text = eventtemplate  # 假设标签是字符串形式
        label_encoding = self.tokenizer(label_text, truncation=True, padding="max_length", max_length=2, return_tensors="pt")

        # 返回编码后的内容和标签
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去除多余的批次维度
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_encoding['input_ids'].squeeze(0)
        }

def load_parsed_bgl(): ##这个函数用于提取训练模版解析的数据集
    parsed_bgl = pd.read_csv('/home/jpy/graduation_design_final/BGL/BGL_2k.log_structured.csv')
    parsed_bgl["Label"] = parsed_bgl["Label"].apply(lambda x: int(x != "-"))
    labels = parsed_bgl['Label'].tolist()
    contents = parsed_bgl['Content'].to_list()
    eventtemplates = parsed_bgl['EventTemplate'].to_list()
    label_content_tuples = list(zip(labels,contents,eventtemplates))

    return label_content_tuples

local_model_path="/home/jpy/graduation_design_final/Flan_T5_base"
new_words = ["<*>", "{", "}", "<", "\\"]
tokenizer = T5Tokenizer.from_pretrained(local_model_path)
tokenizer.add_tokens(new_tokens=new_words)
model = T5ForConditionalGeneration.from_pretrained(local_model_path,device_map="auto")
model.to('cuda')


train_data_tuple=load_parsed_bgl()
traindata=TrainData(train_data_tuple,tokenizer)
dataloader=DataLoader(traindata,5,shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-4)
# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# 训练过程
epochs = 30
loop = tqdm(total=epochs)

for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        # 清除梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        logits = outputs.logits  # 获取 logits
        result_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        # 反向传播和优化
        result_loss.backward()
        optimizer.step()
    loop.update(1)

model_save_path = "/home/jpy/graduation_design_final/Flan_T5_base_tuning"
tokenizer_save_path = "/home/jpy/graduation_design_final/Tokenizer"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)
