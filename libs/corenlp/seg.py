from ..task import Task


class CoreNLPSentenceSegmenter(Task):

    def __init__(self, config):
        print(config)

    def run(self, data):
        print(data)