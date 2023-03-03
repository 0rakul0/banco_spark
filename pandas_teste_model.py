import pandas as pd
import spacy

model_path = "./modelo_textcat"

# Carrega o modelo
nlp = spacy.load(model_path)

# Testa o modelo
text = "O juiz Juiz Paulo Souza julgou procedente o processo de tráfico de drogas na vara 2ª Vara Criminal de Curitiba do foro TJPR"

doc = nlp(text)
if doc.cats['Procedente'] > doc.cats['Improcedente']:
    print(f"{text} é procedente")
else:
    print(f"{text} é improcedente")
