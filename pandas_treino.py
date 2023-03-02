import pandas as pd
import spacy
import random
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
from spacy.lang.pt.stop_words import STOP_WORDS


def load_data(file_path, text_col, label_col, label_map=None):
    """
    Carrega os dados de um arquivo CSV em um DataFrame e retorna uma lista de tuplas (texto, label)
    :param file_path: caminho do arquivo CSV
    :param text_col: nome da coluna que contém o texto
    :param label_col: nome da coluna que contém a label
    :param label_map: dicionário que mapeia as labels originais para as novas labels (opcional)
    :return: lista de tuplas (texto, label)
    """
    df = pd.read_csv(file_path)
    if label_map:
        df[label_col] = df[label_col].replace(label_map)
    return [(text, label) for text, label in zip(df[text_col], df[label_col])]


def train_textcat_model(train_data, model_path, n_iter=10, dropout=0.2, batch_size=16):
    """
    Treina um modelo de classificação de texto do spaCy, salva-o em um arquivo e retorna o caminho do arquivo.
    :param train_data: lista de tuplas (texto, label) de treinamento
    :param model_path: caminho para salvar o modelo treinado
    :param n_iter: número de iterações de treinamento
    :param dropout: taxa de dropout
    :param batch_size: tamanho do minibatch
    :return: caminho do arquivo onde o modelo foi salvo
    """
    nlp = spacy.load('pt_core_news_lg')
    nlp.vocab.add_flag(lambda s: s.lower() in STOP_WORDS, spacy.attrs.IS_STOP)
    nlp = spacy.blank('pt')
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe('textcat')
    labels = set([label for text, label in train_data])
    for label in labels:
        textcat.add_label(label)

    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(batch_size, 4, 1.001))
        losses = {}
        for batch in batches:
            texts, labels = zip(*batch)
            # Aplica a tokenização, stemming e remove as stop words
            docs = [nlp(text.lower()) for text in texts]
            stemmed_docs = []
            for doc in docs:
                stemmed_docs.append(" ".join([token.lemma_ for token in doc if not token.is_stop]))
            golds = [{"cats": {label: True}} for label in labels]
            nlp.update(stemmed_docs, golds, sgd=optimizer, drop=dropout, losses=losses)

    # Salva o modelo
    nlp.to_disk(model_path)
    return model_path


# Exemplo de uso
file_path = "./data/sentencas.csv"
text_col = 'texto_sentenca'
label_col = 'sentenca'
label_map = {'label_original': 'nova_label'}
train_data = load_data(file_path, text_col, label_col, label_map=label_map)
model_path = "modelo_textcat"

# Treina o modelo e salva-o em um arquivo
n_iter = 10
dropout = 0.2
batch_size = 16
train_textcat_model(train_data, model_path, n_iter=n_iter, dropout=dropout, batch_size=batch_size)

# Carrega o modelo
nlp = spacy.load(model_path)

# Testa o modelo
text = "O réu nega ter tido qualquer envolvimento com o crime."
doc = nlp(text)
if doc.cats['Procedente'] > doc.cats['Improcedente']:
    print(f"{text} é procedente")
else:
    print(f"{text} é improcedente")
