import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
from transformers import EvalPrediction

def carrega_modelo(pasta):
    
    if not os.path.isdir(pasta):
        print(f"A pasta {pasta} não foi encontrada.")
        return
    
    print(f"Carregando modelo: {pasta}")

    labels_path = os.path.join(pasta, 'num_labels.txt')
    multilabel_path = os.path.join(pasta, 'multilabel.pkl')
    with open(labels_path, 'r') as f:
        num_labels = int(f.read().strip())

    tokenizer = DistilBertTokenizer.from_pretrained(pasta)
    modelo = DistilBertForSequenceClassification.from_pretrained(pasta, num_labels=num_labels, problem_type="multi_label_classification")

    with open(multilabel_path, 'rb') as f:
        multilabel = pickle.load(f)

    print(f'Modelo carregado com sucesso.')

    return modelo, tokenizer, multilabel

def salva_modelo(modelo, num_labels, multilabel, tokenizer, pasta):
    os.makedirs(pasta, exist_ok=True)

    #model_path = os.path.join(pasta, 'model.pth')
    labels_path = os.path.join(pasta, 'num_labels.txt')
    multilabel_path = os.path.join(pasta, 'multilabel.pkl')

    modelo.save_pretrained(pasta)
    #torch.save(modelo.state_dict(), model_path)

    tokenizer.save_pretrained(pasta)

    with open(labels_path, 'w') as f:
        f.write(str(num_labels))

    with open(multilabel_path, 'wb') as f:
        pickle.dump(multilabel, f)

    print(f'Modelo salvo em: {pasta}')

def loop_modelo(modelo, tokenizer, multilabel):
    while True:
        texto = input('Digite um texto ou "sair" para encerrar: \n')
        if texto == 'sair':
            break
        else:
            print(analisa_prompt(modelo, tokenizer, multilabel, texto))

# Funções adaptadas de: https://github.com/laxmimerit/NLP-Tutorials-with-HuggingFace/
def analisa_prompt(modelo, tokenizer, multilabel, texto):

    encoding = tokenizer(texto, return_tensors='pt')
    encoding.to(modelo.device)

    outputs = modelo(**encoding)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits[0].cpu())
    preds = np.zeros(probs.shape)
    preds[np.where(probs>=0.3)] = 1

    multilabel.classes_

    return multilabel.inverse_transform(preds.reshape(1,-1))

class distilbertDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_len=128):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    label = torch.tensor(self.labels[idx])

    encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': label
    }
  
def multi_labels_metrics(predictions, labels, threshold=0.3):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    hamming = hamming_loss(y_true, y_pred)

    metrics = {
        "hamming_loss": hamming,
        "f1": f1
    }

    return metrics

def compute_metrics(p:EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    result = multi_labels_metrics(predictions=preds,
                                labels=p.label_ids)

    return result