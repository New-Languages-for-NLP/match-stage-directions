import streamlit as st
from streamlit import sidebar as sb
from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

@st.cache
def load_data():

    dataset = pd.read_csv('stage_directions.csv')

    seed = 38383
    dataset = dataset.sample(frac=1,random_state=seed).reset_index(drop=True)

    true_ = dataset[dataset.stage==1]
    false_ = dataset[dataset.stage==0]
    false_subsample = false_[:len(true_)]

    dataset = pd.concat([true_, false_subsample])

    dataset = dataset.sample(frac=1,random_state=seed).reset_index(drop=True)

    return dataset


options = {
    "ORTH": partial(sb.text_input, "ORTH"),
    "TEXT": partial(sb.text_input, "TEXT"),
    "LOWER": partial(sb.text_input, "LOWER"),
    "LENGTH": partial(sb.number_input, "LENGTH"),
    "IS_ALPHA": partial(sb.checkbox, "IS_ALPHA"),
    "IS_ASCII": partial(sb.checkbox, "IS_ASCII"),
    "IS_DIGIT": partial(sb.checkbox, "IS_DIGIT"),
    "IS_LOWER": partial(sb.checkbox, "IS_LOWER"),
    "IS_UPPER": partial(sb.checkbox, "IS_UPPER"),
    "IS_TITLE": partial(sb.checkbox, "IS_TITLE"),
    "IS_PUNCT": partial(sb.checkbox, "IS_PUNCT"),
    "IS_SPACE": partial(sb.checkbox, "IS_SPACE"),
    "IS_STOP": partial(sb.checkbox, "IS_STOP"),
    "IS_SENT_START": partial(sb.checkbox, "IS_SENT_START"),
    "LIKE_NUM": partial(sb.checkbox, "LIKE_NUM"),
    "LIKE_URL": partial(sb.checkbox, "LIKE_URL"),
    "LIKE_EMAIL": partial(sb.checkbox, "LIKE_EMAIL"),
    "SPACY": partial(sb.checkbox, "SPACY"),
    "POS": partial(sb.text_input, "POS"),
    "TAG": partial(sb.text_input, "TAG"),
    "MORPH": partial(sb.text_input, "MORPH"),
    "DEP": partial(sb.text_input, "DEP"),
    "LEMMA": partial(sb.text_input, "LEMMA"),
    "SHAPE": partial(sb.text_input, "SHAPE"),
    "ENT_TYPE": partial(sb.text_input, "ENT_TYPE"),
    "_": partial(sb.text_input, "_"),
    "OP": partial(sb.selectbox, "OP", ["!", "?", "+", "*"]),
}

patterns = []
n_patterns = sb.number_input('number of patterns', 1)

for ipattern in range(n_patterns):
    sb.write(f"## {ipattern}. pattern:")

    n_tokens = sb.number_input("number of tokens", 1, key=f"{ipattern}")
    patterns.append([])
    
    for itoken in range(n_tokens):
        sb.write(f"### {ipattern}-{itoken}. token:")
        n_attributes = sb.number_input("number of attributes", 1, key=f"{ipattern}-{itoken}")
        patterns[-1].append({})
        for iattribute in range(n_attributes):
            key=f"{ipattern}-{itoken}-{iattribute}"
            attribute_type = sb.selectbox("type", list(options.keys()), key=key)
            attribute_value = options[attribute_type]('Enter', key=key)
            patterns[-1][-1][attribute_type] = attribute_value

st.write("## Patterns")
for pattern in patterns:
    st.code(pattern)

import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

matcher.add("StageDirections", patterns)

y_true = []
y_pred = []
dataset = load_data()
for irow, row in dataset.iterrows():
    text, label = row[['text', 'stage']]

    doc = nlp(text)
    matches = matcher(doc)
    y_pred.append(len(matches) > 0)

    y_true.append(label)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
st.write("## Classification report")
st.text(classification_report(y_true, y_pred))

st.write("## Hits and misses")

n_lines_to_print = st.number_input("How many lines to print?", 1, value=3)
st.write('`true`, `predicted`, `text`')
for irow, (t,p,text) in enumerate(zip(y_true, y_pred, dataset.text)):
    if irow>=n_lines_to_print:
        break
    st.write(t, p, text)
