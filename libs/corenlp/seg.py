from ..task import Task
import stanfordnlp

class CoreNLPSentenceSegmenter(Task):

    def __init__(self, config):
        self.config = config

        # Download english model
        # stanfordnlp.download('en','venv/share',confirm_if_exists=True)

        # Specify the local dir of the model and pipeline
        self.nlp = stanfordnlp.Pipeline(lang='en', models_dir='venv/share', processors='tokenize')

    def run(self, data):
        paragraphs = []
        words = 0

        for paragraph in data:
            sentences = []
            doc = self.nlp(paragraph)

            for sentence in doc.sentences:
                sentences.append(sentence.words)
                words += len(sentence.words)

            paragraphs.append(sentences)

        return paragraphs, words
