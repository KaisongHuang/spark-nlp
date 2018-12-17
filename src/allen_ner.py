def setup():

    from allennlp.predictors.predictor import Predictor
    from spacy import load

    segmenter = load("en")
    segmenter.disable_pipes("parser", "tagger", "ner")
    segmenter.add_pipe(segmenter.create_pipe("sentencizer"))
    
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz")

    return segmenter, predictor

def ner(nlp, predictor, docs):

    results = []
    words = 0

    # For each parsed doc...
    for parsed_doc in nlp.pipe(docs):

        # Keep track of sentences in the paragraph
        par = []

        # For each sentence in the paragraph, fetch the entities
        for sentence in parsed_doc.sents:
            words += len(sentence)
            pred = predictor.predict(sentence=sentence.text)
            par.append({
                "text": pred["words"],
                "entities": pred["tags"]
            })
        
        results.append(par)

    return results, words