from ..task import Task


class AllenNLPNamedEntityRecognition(Task):

    def __init__(self, gpu):
        print(gpu)

    def run(self, data):
        print(data)
