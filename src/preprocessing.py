import spacy

_nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

NEGATIONS = {"ne", "n", "pas", "jamais", "rien", "aucun", "plus", "non", "sans"}


def tokeniser(texte: str) -> list[str]:
    doc = _nlp(texte.lower())
    tokens = []
    for t in doc:
        if not t.is_alpha:
            continue
        # on garde les négations, on retire les autres stop words
        if t.is_stop and t.lemma_ not in NEGATIONS and t.text not in NEGATIONS:
            continue
        tokens.append(t.lemma_)
    return tokens
