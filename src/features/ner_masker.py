"""
NER-based topic masking for SBERT inputs.

Replaces named entities with their type labels to force the model
to rely on writing style rather than content/topic keywords.
Based on TDRLM (Expert Systems with Applications, 2023).
"""

import spacy

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


def mask_entities(text: str) -> str:
    nlp = _get_nlp()
    doc = nlp(text)
    if not doc.ents:
        return text
    parts = []
    prev_end = 0
    for ent in doc.ents:
        parts.append(text[prev_end:ent.start_char])
        parts.append(f"<{ent.label_}>")
        prev_end = ent.end_char
    parts.append(text[prev_end:])
    return "".join(parts)


def mask_document_sentences(sentences: list) -> list:
    return [mask_entities(s) for s in sentences]