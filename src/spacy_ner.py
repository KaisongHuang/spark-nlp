import os

# Called to enable GPU acceleration
def setup(gpu):
    
    import spacy

    # Should we use a GPU?
    if gpu >= 0:
        # Set GPU env. variable
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Tell spacy to use the GPU
        spacy.require_gpu()

    # Load the English model
    return spacy.load("en", disable=["tagger"])


# Given a document (list of paragraphs), perform NER on each sentence
def ner(nlp, docs):

    results = []
    words = 0

    # For each parsed doc...
    for doc in nlp.pipe(docs):

        # Keep track of sentences in the paragraph
        par = []

        # For each sentence in the paragraph, fetch the entities
        for sentence in doc.sents:
            words += len(sentence)
            par.append({
                "text": sentence,
                "entities": get_entities(sentence)
            })

        # Add the paragraph to our results
        results.append(par)

    return results, words


# Get the entities in a sentence
def get_entities(sent):
    return [{e.text: (e.label_, e.start_char - e.sent.start_char, e.end_char - e.sent.start_char)} for e in sent.ents]
