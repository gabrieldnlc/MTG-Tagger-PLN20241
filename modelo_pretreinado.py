import argparse
from utils import carrega_modelo, loop_modelo


def main():

    parser = argparse.ArgumentParser(description='Carrega o distilBERT com fine-tuning de uma pasta local.')
    parser.add_argument('folder', type=str, help='A pasta onde o modelo foi salvo.')
    args = parser.parse_args()
    pasta = args.folder

    modelo, tokenizer, multilabel = carrega_modelo(pasta)

    loop_modelo(modelo, tokenizer, multilabel)
    
if __name__ == "__main__":
    main()