from ..task import Task

class CoreNLPDependencyParsing(Task):

    def __init__(self, spark_context):
        self.sc = spark_context

    def run(self, data):
        print()
