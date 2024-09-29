Projeto referente à disciplina de Processamento de Linguagem Natural, UFCG 2024.1

- Gabriel Dantas Lima Costa 122210429
- Paulo Ricardo Oliveira de Macêdo 121110379

INSTALAÇÃO:
```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt
```

Após isso, para testar o modelo pré-treinado disponível:

Primeiro, baixe-o em https://drive.google.com/file/d/1FErW9JjVyxwqoc6OiqzUOJ2kQbkXdx0n (arquivo muito grande para o Git -- 250 MB)

Extraia-o na pasta raíz do projeto. Então:
```bash
python modelo_pretreinado.py pretreinado
```

Para treinar o seu próprio modelo:
```bash
python treina_modelo.py <dataset em .csv>
python modelo_pretreinado.py <pasta do modelo (padrão: 'modelo')>
```
