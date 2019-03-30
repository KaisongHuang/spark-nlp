from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from spacy.lang.en import English
from ..task import Task
import torch
torch.backends.cudnn.enabled = False


class AllenNLPNamedEntityRecognition(Task):
    def __init__(self, gpu):
        self.nlp = English()
        self.sentencizer = self.nlp.create_pipe("sentencizer")
        self.nlp.add_pipe(self.sentencizer)
        archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz", cuda_device = int(gpu))
        self.predictor = Predictor.from_archive(archive)


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
                prd_pos = prediction['tags']
                length = len(prd_words)
                
                for index in range(length):
                    pair = (prd_words[index], prd_pos[index])
                    sent.append(pair)
                
                words += length
                par.append(sent)
            
            results.append(par)

        return results, words
