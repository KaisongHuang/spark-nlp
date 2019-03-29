from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
import spacy
from spacy.lang.en import English

from ..task import Task


class AllenNLPPartOfSpeechTagger(Task):
    def __init__(self, gpu):
        self.nlp = English()
        self.sentencizer = self.nlp.create_pipe("sentencizer")
        self.nlp.add_pipe(self.sentencizer)
        archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz", cuda_device = int(gpu))
        self.predictor = Predictor.from_archive(archive)
    
    # The parameter data here is an article which is a list of paragraphs
    def run(self, data):
        results = []
        words = 0

        for paragraph in data:
            par = []
            
            doc = self.nlp(paragraph)
            for sentence in doc.sents:
                sent = []
                prediction = self.predictor.predict(str(sentence))
                prd_words = prediction['words']
                prd_pos = prediction['pos']
                length = len(prd_words)
                
                for index in range(length):
                    pair = (prd_words[index], prd_pos[index])
                    sent.append(pair)
                
                words += length
                par.append(sent)
            
            results.append(par)

        return results, words
