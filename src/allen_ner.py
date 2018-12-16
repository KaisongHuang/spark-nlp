def setup():
    from allennlp.predictors.predictor import Predictor
    from spacy import load
    return load("en", disable=["tagger", "ner"]), Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz")


def ner(nlp, predictor, doc):

    results = []

    # For each parsed doc...
    for parsed_doc in nlp.pipe(doc):

        # Keep track of sentences in the paragraph
        par = []

        # For each sentence in the paragraph, fetch the entities
        for sentence in parsed_doc.sents:
            par.append(predictor.predict(sentence=sentence))
            # par.append({
            #     "text": sentence,
            #     "entities": get_entities(sentence)
            # })

        # Add the paragraph to our results
        results.append(par)

    return results
