from ..task import Task
from allennlp.predictors.predictor import Predictor
from spacy import load


class AllenNLPDependencyParsing(Task):

    def __init__(self, gpu):
        self.gpu = gpu
        self.nlp = load("en", disable=["tagger", "ner"])
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    def run(self, data):
        results = []
        words = 0

        # For each parsed doc...
        for doc in self.nlp.pipe(data):

            par = []

            batch = [{"sentence": sentence.text} for sentence in doc.sents]

            try:
                predictions = self.predictor.predict_batch_json(batch)
            except:
                continue

            for prediction in predictions:
                words += len(prediction["words"])
                par.append({
                    "text": prediction["words"],
                    "dep": prediction["predicted_dependencies"]
                })

            results.append(par)

        return results, words