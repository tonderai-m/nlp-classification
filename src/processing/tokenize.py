import numpy as np
import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = None
        self.raw_text = dataset['feature']
        self.create_encoding()
        self.labels = None
        self.raw_labels = dataset['target']
        self.create_target()
        self.input_ids = self.encodings['input_ids']
        self.attention_mask = self.encodings['attention_mask']
        self.output_len = len(self.labels[0])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def create_encoding(self):
        x = self.tokenizer(text=self.raw_text.tolist(),
                           add_special_tokens=True,
                           max_length=self.max_len,
                           padding='max_length',
                           return_tensors='pt',
                           return_token_type_ids=False,
                           return_attention_mask=True,
                           verbose=True)
        self.encodings = x

    def create_target(self):
        y = self.raw_labels
        y = torch.tensor(np.stack(y.values))
        self.labels = y
