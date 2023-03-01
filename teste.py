from pyspark.shell import spark
from pyspark.sql.functions import to_date
from pyspark.sql.functions import datediff
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS

print(spark.version)

df_data = spark.read.csv(path="./data/dados_juridicos.csv", header=True)

dados_npl = spacy.load('pt_core_news_lg')

df_data = df_data.withColumn("data_inicio_processo", to_date("data_inicio_processo", "yyyy-MM-dd"))
df_data = df_data.withColumn("data_fim_processo", to_date("data_fim_processo", "yyyy-MM-dd"))
df_data = df_data.withColumn("dias_corridos", datediff("data_fim_processo", "data_inicio_processo"))

df_data = df_data.select("classe", "assunto", "vara", "foro", "juiz", "data_inicio_processo", "data_fim_processo", "dias_corridos", "sentenca", "acolhido", "recurso")

print(df_data.show(5))