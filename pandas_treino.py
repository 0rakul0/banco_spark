import pandas as pd
import spacy
import numpy as np
import random
from spacy.util import minibatch, compounding
import seaborn as sns
import matplotlib.pyplot as plt
from util.stop_words import STOP_WORDS


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


def train_textcat_model(train_data, model_path, n_iter=10, dropout=0.2, batch_size=16, lemmatize=True, remove_stopwords=True):
    """
    Treina um modelo de classificação de texto do spaCy, salva-o em um arquivo e retorna o caminho do arquivo.
    :param train_data: lista de tuplas (texto, label) de treinamento
    :param model_path: caminho para salvar o modelo treinado
    :param n_iter: número de iterações de treinamento
    :param dropout: taxa de dropout
    :param batch_size: tamanho do minibatch
    :param lemmatize: se True, aplica lematização nos textos
    :param remove_stopwords: se True, remove as stop words dos textos
    :return: caminho do arquivo onde o modelo foi salvo
    """
    global golds
    nlp = spacy.load('pt_core_news_lg')
    if remove_stopwords:
        nlp.vocab.add_flag(lambda s: s.lower() in STOP_WORDS, spacy.attrs.IS_STOP)
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
            # Aplica a tokenização e lematização (se necessário)
            docs = []
            for text in texts:
                doc = nlp(text.lower())

                # remove as stop_words
                if remove_stopwords:
                    doc = [token for token in doc if not token.is_stop]

                # deixa o radical da frase
                if lemmatize:
                    doc = [token.lemma_ for token in doc]
                else:
                    doc = [token.text for token in doc]

                # junta os textos
                docs.append(" ".join(doc))

            golds = [{"cats": {label: True}} for label in labels]

            nlp.update(docs, golds, sgd=optimizer, drop=dropout, losses=losses)

    # Salva o modelo
    nlp.to_disk(model_path)
    return model_path


# Define a função para gerar o mapa de calor
def plot_textcat_heatmap(doc, decisao):
    doc = [token for token in doc if not token.is_stop]
    data = []
    for token in doc:
        row = [token.text] + [token.vector[i] for i in range(len(token.vector))]
        data.append(row)
    df = pd.DataFrame(data, columns=['token'] + [f'vec_{i}' for i in range(len(token.vector))])
    df = pd.melt(df, id_vars=['token'], value_vars=[f'vec_{i}' for i in range(len(token.vector))], var_name='dim',
                 value_name='value')
    df['abs_value'] = df['value'].abs()
    pivot_table = df.pivot_table(index='token', columns='dim', values='abs_value', aggfunc='max')
    pivot_table = pivot_table.loc[:, (pivot_table.mean() > pivot_table.mean().mean())]
    pivot_table_grouped = pivot_table.groupby(np.arange(len(pivot_table.columns)) // 10, axis=1).max()
    if pivot_table.size > 0:
        ax = sns.heatmap(pivot_table_grouped, cmap='Reds', xticklabels=True, yticklabels=True, )
        ax.set_title = decisao
        file_name = str(doc).replace(' ','_').replace('.','').replace(',', '').replace('[','').replace(']','')
        plt.savefig(f'./img/{file_name}_{decisao}.png')
    else:
        print("A matriz de frequência está vazia.")


# Exemplo de uso
file_path = "./data/sentencas.csv"
text_col = 'texto_sentenca'
label_col = 'sentenca'
label_map = {'label_original': 'nova_label'}
train_data = load_data(file_path, text_col, label_col, label_map=label_map)
model_path = "./modelo_textcat"

# Treina o modelo e salva-o em um arquivo
n_iter = 10
dropout = 0.2
batch_size = 16
train_textcat_model(train_data, model_path, n_iter=n_iter, dropout=dropout, batch_size=batch_size)

# Carrega o modelo
nlp = spacy.load(model_path)

# Testa o modelo
text = "O réu admitiu ter cometido o crime."
doc = nlp(text)
score = doc.cats
if doc.cats['Procedente'] > doc.cats['Improcedente']:
    decisao = 'procedente'
    print(f"{text} é procedente")
else:
    decisao = 'improcedente'
    print(f"{text} é improcedente")

plot_textcat_heatmap(doc, decisao)
plt.show()
