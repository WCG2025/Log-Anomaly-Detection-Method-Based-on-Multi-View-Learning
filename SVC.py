import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

from thundersvm import NuSVC
# import sys
# print(sys.executable)
import json
import numpy as np

train_data_file='/root/autodl-tmp/V100/data/train_svc.json'
test_data_file='/root/autodl-tmp/V100/data/test_svc.json'

with open(train_data_file, "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_values_list = [entry["values"] for entry in train_data] 
train_label_list=[entry['label'] for entry in train_data]

train_values_array=np.array(train_values_list,dtype=np.float64)
train_label_array=np.array(train_label_list,dtype=int)

print("start training")

from thundersvm import SVC
clf=SVC(C=50)
clf.fit(train_values_array,train_label_array)

print("training finish")
with open(test_data_file,"r",encoding="utf-8") as f:
    test_data= json.load(f)

test_values_list=[entry['values'] for entry in test_data]
test_label_list=[entry['label'] for entry in test_data]

test_values_array=np.array(test_values_list,dtype=np.float64)
test_label_array=np.array(test_label_list,dtype=int)

print("start testing")

predict_labels=clf.predict(test_values_array)

TP=0
FP=0
TN=0
FN=0
print("start saving")
np.save("/root/autodl-tmp/V100/data/predict_array.npy",predict_labels)
print("saving finish")


for i in range(len(predict_labels)):
    if test_label_array[i]==0 and predict_labels[i]==0:
        TN+=1
    if test_label_array[i]==0 and predict_labels[i]==1:
        FP+=1
    if test_label_array[i]==1 and predict_labels[i]==1:
        TP+=1
    if test_label_array[i]==1 and predict_labels[i]==0:
        FN+=1

accuracy= (TP+TN)/(TP+TN+FP+FN)
precision= TP/(TP+FP) if (TP + FP) != 0 else 0
Recall= TP/(TP+FN) if (TP + FP) != 0 else 0
score=(2*precision*Recall)/(precision+Recall)


eval_data = {
    "TP":TP,
    "TN":TN,
    "FP":FP,
    "FN":FN,
    "accuracy":accuracy,
    "precision": precision,
    "recall": Recall,
    "f1_score": score
}

# 写入文件
with open('/root/autodl-tmp/V100/eval.json', 'w') as f:
    json.dump(eval_data, f, indent=4)

print("testing finish")