import os

import spacy


# Called to enable GPU acceleration
def setup(gpu):
    # Should we use a GPU?
    if gpu >= 0:
        # Set GPU env. variable
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Tell spacy to use the GPU
        spacy.require_gpu()

    # Load the English model
    nlp = spacy.load("en")

    return nlp


# Given a document (list of paragraphs), perform NER on each sentence
def ner(nlp, doc):
    results = []

    # For each parsed doc...
    for parsed_doc in nlp.pipe(doc):

        # Keep track of sentences in the paragraph
        par = []

        # For each sentence in the paragraph, fetch the entities
        for sentence in parsed_doc.sents:
            par.append({
                "text": sentence,
                "entities": get_entities(sentence)
            })

        # Add the paragraph to our results
        results.append(par)

    return results


# Get the entities in a sentence
def get_entities(sentence):
    return [{ent.text: (ent.label_, ent.start_char - ent.sent.start_char, ent.end_char - ent.sent.start_char)} for ent in sentence.ents]