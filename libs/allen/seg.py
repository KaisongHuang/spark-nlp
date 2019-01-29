from ..task import Task


class AllenNLPSentenceSegmenter(Task):

    def __init__(self, config):
        print(config)

    def run(self, data):
        print(data)
