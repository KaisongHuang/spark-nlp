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
def get_entities(sent):
    return [{e.text: (e.label_, e.start_char - e.sent.start_char, e.end_char - e.sent.start_char)} for e in sent.ents]
