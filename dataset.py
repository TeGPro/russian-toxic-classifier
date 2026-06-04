import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class ToxicDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128, batch_size=1024):
        self.labels = hf_dataset['label']
        texts = [str(t) for t in hf_dataset['text']]

        input_ids_list = []
        attention_mask_list = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Токенизация датасета"):
            batch_texts = texts[i : i + batch_size]

            encoding = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids_list.append(encoding['input_ids'])
            attention_mask_list.append(encoding['attention_mask'])

        self.input_ids = torch.cat(input_ids_list, dim=0)
        self.attention_mask = torch.cat(attention_mask_list, dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }