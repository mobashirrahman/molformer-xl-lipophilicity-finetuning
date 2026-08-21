"""
Shared dataset and model classes used by finetune_lipophilicity.py, task2.py, and task3.py.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LipoDataset(Dataset):
    def __init__(self, smiles_list, targets, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        encoding = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(target, dtype=torch.float)
        return item


class MolFormerRegressor(nn.Module):
    def __init__(self, base_model, dropout_rate=0.0):
        super(MolFormerRegressor, self).__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        regression_logits = self.regressor(pooled_output)
        return regression_logits
