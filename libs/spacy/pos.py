from ..task import Task


class SpacyPartOfSpeechTagger(Task):

    def __init__(self, config):
        print(config)

    def run(self, data):
        print(data)
