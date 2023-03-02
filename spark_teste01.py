from pyspark.shell import spark
from pyspark.sql.functions import to_date, datediff, col, rand
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
import spacy
import random
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
from spacy.lang.pt.stop_words import STOP_WORDS

df_data = spark.read.csv(path="./data/dados_juridicos_com_texto_sentenca.csv", header=True)

dados_npl = spacy.load('pt_core_news_lg')

df_data = df_data.withColumn("data_inicio_processo", to_date("data_inicio_processo", "yyyy-MM-dd"))
df_data = df_data.withColumn("data_fim_processo", to_date("data_fim_processo", "yyyy-MM-dd"))
df_data = df_data.withColumn("dias_corridos", datediff("data_fim_processo", "data_inicio_processo"))

df_data = df_data.select("classe", "assunto", "vara", "foro", "juiz", "data_inicio_processo", "data_fim_processo", "dias_corridos","texto_sentenca","sentenca", "acolhido", "recurso")

df_data_sentencas = df_data.select('texto_sentenca', 'sentenca')

df_pandas = df_data_sentencas.toPandas()

# Carrega o modelo Spacy
nlp = spacy.load('pt_core_news_lg')

# Exemplo de função para converter o DataFrame Pandas em TrainingData

def create_training_data(df):
    # Cria uma lista para armazenar as tuplas (doc, gold) correspondentes a cada exemplo
    examples = []
    # Loop pelos dados do DataFrame Pandas
    for index, row in df.iterrows():
        # Obtém o texto da sentença e a label correspondente
        text = row['texto_sentenca']
        label = row['sentenca']
        # Cria um objeto Doc com o texto usando o modelo Spacy
        doc = nlp(text)
        # Cria um objeto GoldParse com as anotações correspondentes
        gold = GoldParse(doc, cats={'SENTENCA': label})
        # Adiciona a tupla (doc, gold) à lista de exemplos
        examples.append((doc, gold))
    # Verifica se o número de exemplos gerados corresponde ao número de documentos lidos
    assert len(examples) == df.shape[0]
    # Retorna a lista de exemplos
    return examples


# Chama a função para criar o TrainingData
examples, num_labels, label_map = create_training_data(df_pandas)

# Define as configurações do modelo
n_iter = 10
dropout = 0.2
batch_size = 16

# Inicia o treinamento do modelo
nlp = spacy.blank('pt')
if 'textcat' not in nlp.pipe_names:
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)
else:
    textcat = nlp.get_pipe('textcat')
for label in range(num_labels):
    textcat.add_label(str(label))
optimizer = nlp.begin_training()
for i in range(n_iter):
    random.shuffle(examples)
    losses = {}
    batches = minibatch(examples, size=batch_size)
    for batch in batches:
        docs, golds = zip(*batch)
        nlp.update(docs, golds, sgd=optimizer, drop=dropout, losses=losses)
        data = list(zip(*batch))
        texts = data[0]
        annotations = data[1:]
        # Converte as labels em um formato que pode ser usado pelo modelo
        gold_labels = [{'cats': {label: True}} for label in annotations]
        # Atualiza o modelo com os dados do minibatch
        nlp.update(texts, gold_labels, sgd=optimizer, drop=dropout, losses=losses)
    print('Losses', losses)
