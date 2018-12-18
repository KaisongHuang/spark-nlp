def setup():

    from allennlp.predictors.predictor import Predictor
    from spacy import load

    segmenter = load("en", disable=["tagger", "ner"])
    
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz")

    return segmenter, predictor

def ner(nlp, predictor, docs):

    results = []
    words = 0

    # For each parsed doc...
    for doc in nlp.pipe(docs):

        par = []

        batch = [{"sentence": sentence.text} for sentence in doc.sents]

        try:
            predictions = predictor.predict_batch_json(batch)
        except:
            continue

        for prediction in predictions:
            words += len(prediction["words"])
            par.append({
                "text": prediction["words"],
                "entities": prediction["tags"]
            })

        results.append(par)

    return results, words
