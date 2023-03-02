import pandas as pd

# carrega o arquivo CSV
dados_juridicos = pd.read_csv('./data/dados_juridicos.csv')

# cria uma nova coluna 'texto_sentenca' e preenche com o texto da sentença correspondente
dados_juridicos['texto_sentenca'] = dados_juridicos.apply(lambda row: f'O juiz {row["juiz"]} julgou {row["sentenca"].lower()} o processo de {row["assunto"].lower()} na vara {row["vara"]} do foro {row["foro"]}.', axis=1)

# salva o arquivo CSV com a nova coluna
dados_juridicos.to_csv('./data/dados_juridicos_com_texto_sentenca.csv', index=False)
