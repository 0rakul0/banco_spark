import pandas as pd
import spacy

model_path = "./modelo_textcat"

# Carrega o modelo
nlp = spacy.load(model_path)

# Testa o modelo
text = "O juiz condenou o réu à prisão perpétua."

doc = nlp(text)
if doc.cats['Procedente'] > doc.cats['Improcedente']:
    print(f"{text} é procedente")
else:
    print(f"{text} é improcedente")
