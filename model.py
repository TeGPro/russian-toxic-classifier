import torch.nn as nn
from transformers import AutoModel

class BertClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
        self.classifier = nn.Linear(312, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits