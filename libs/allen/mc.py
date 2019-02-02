from allennlp.predictors.predictor import Predictor
from ..task import Task


class AllenNLPMachineComprehension(Task):
    def __init__(self,gpu,question):
        self.gpu = gpu
        self.question = question
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")

    def run(self, data):
        result = []
        words = 0

        # concatenate paragraphs
        passage = ''
        for paragraph in data:
            passage += paragraph

        # make predictions based on the passage
        prediction = self.predictor.predict(passage=passage, question=self.question)
        words += len(prediction["passage_tokens"])
        result.append({
            "passage": passage,
            "question": self.question,
            "answer": prediction["best_span_str"]
        })

        return result, words