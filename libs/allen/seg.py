from ..task import Task


class AllenNLPSentenceSegmenter(Task):

    def __init__(self, gpu):
        print(gpu)

    def run(self, data):
        print(data)
