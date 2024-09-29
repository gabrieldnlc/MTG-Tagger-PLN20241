import argparse
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from utils import distilbertDataset, compute_metrics, salva_modelo
import numpy as np
import json

def main():

    # Captura argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Treina o distilBERT com um dataset.')
    parser.add_argument('csv', type=str, help='O dataset, em formato .csv', default='dataset.csv', nargs='?')
    args = parser.parse_args()
    dataset = args.csv
    
    # Carrega o dataset
    df = pd.read_csv(dataset, usecols=["name", "tags", "oracle_text"])

    df.info()

    df.duplicated().sum()

    # No CSV, as tags são armazenadas como uma grande string separada por vírgula
    df['tags'] = df['tags'].str.split(',')

    # Para fins de análise, imprime as tags mais comuns no dataset escolhido
    tag_counts = [tag for tags in df['tags'] for tag in tags]
    print(pd.Series(tag_counts).value_counts())

    # Transformando as tags em vetores one-hot encoding
    multilabel = MultiLabelBinarizer()
    tags = multilabel.fit_transform(df['tags']).astype('float32')
    descricoes = df['oracle_text'].tolist()

    # Separação do set de testes e o set de validação: 80% treino, 20% validação
    texto_treino, texto_validacao, tags_treino, tags_validacao = train_test_split(descricoes, tags,
                                                                    test_size=0.2, random_state=50)
    
    # O modelo utilizado é o distilbert, um modelo mais leve que o BERT original
    # Leitura recomendada para entender: https://medium.com/huggingface/distilbert-8cf3380435b5
    checkpoint = "distilbert-base-uncased" # Ignora a caixa alta
    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
    num_labels = len(tags[0])

    # O modelo precisa ser inicializado com problem_type == multi_label_classification
    # para que a função de perda seja apropriada para o problema de classificação multilabel
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels,
                                                                problem_type="multi_label_classification")
    
    dataset_treino = distilbertDataset(texto_treino, tags_treino, tokenizer)
    dataset_validacao = distilbertDataset(texto_validacao, tags_validacao, tokenizer)

    # Argumentos para o processo de treinamento:
    tamanho_lotes = 8 # Lotes menores == menos memória, mais tempo
    epocas = 5 # Quantas vezes o modelo percorrerá o dataset inteiro. Mais épocas, mais tempo.
    checkpoint = 1000 # Salva o modelo a cada 1000 passos.
    num_checkpoints = 3 # Salva até 3 checkpoints, sobreescrevendo os antigos.

    args = TrainingArguments(
    per_device_train_batch_size=tamanho_lotes,
    per_device_eval_batch_size=tamanho_lotes,
    output_dir = './metricas',
    num_train_epochs=epocas,
    save_steps=checkpoint,
    save_total_limit=num_checkpoints
        )

    trainer = Trainer(model=model,
                    args=args,
                    train_dataset=dataset_treino,
                    eval_dataset = dataset_validacao,
                    compute_metrics=compute_metrics)
    trainer.train()

    estatisticas = trainer.evaluate()
    print(estatisticas) 
    with open("estatisticas.json", "w") as outfile:
        json.dump(estatisticas, outfile)

    carta_teste = r'Draw three cards.'

    encoding = tokenizer(carta_teste, return_tensors='pt')
    encoding.to(trainer.model.device)

    outputs = trainer.model(**encoding)

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits[0].cpu())
    preds = np.zeros(probs.shape)
    preds[np.where(probs>=0.3)] = 1

    multilabel.classes_
    print("Carta teste: ", carta_teste)
    print(multilabel.inverse_transform(preds.reshape(1,-1)))

    salva_modelo(trainer.model, num_labels, multilabel, tokenizer, 'modelo')

    

if __name__ == "__main__":
    main()