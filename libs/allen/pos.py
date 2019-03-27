from allennlp.predictors.predictor import Predictor
import spacy
from spacy.lang.en import English

from ..task import Task


class AllenNLPPartOfSpeechTagger(Task):
    def __init__(self, gpu):
        self.gpu = gpu
        self.nlp = English()
        self.sentencizer = self.nlp.create_pipe("sentencizer")
        self.nlp.add_pipe(self.sentencizer)
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    
    # The parameter data here is an article which is a list of paragraphs
    def run(self, data):
        results = []
        words = 0

        for paragraph in data:
            par = []
            
            sent_pos = []
            doc = self.nlp(paragraph)
            for sentence in doc.sents:
                prediction = self.predictor.predict(str(sentence))
                sent_pos.append(prediction['words'])
                sent_pos.append(prediction['pos'])
                words += len(prediction['words'])

            par.append(sent_pos)
            results.append(par)

        return results, words
