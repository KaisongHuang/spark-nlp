from ..task import Task

class CoreNLPDependencyParsing(Task):

    def __init__(self, spark_context):
        self.sc = spark_context

        # Get an instance of the Properties
        self.props = self.sc._jvm.java.util.Properties()
        self.setProperty("annotators", "tokenize,ssplit,pos")
        self.setProperty("ner.useSUTime", "false")

        # Get an instance of the Pipeline
        self.pipeline = self.sc._jvm.edu.stanford.nlp.pipeline.StanfordCoreNLP(self.props)

    def run(self, data):
        results = []
        words = 0

        # For each parsed doc...
        for paragraph in data:

            # Keep track of sentences in the paragraph
            par = []

            doc = self.sc._jvm.edu.stanford.nlp.pipeline.CoreDocument(paragraph)
            self.pipeline.annotate(doc)

            # For each sentence in the paragraph, parse the dependency
            for sentence in doc.sentences():
                words += len(sentence)

                par.append({
                    "text": sentence,
                    "dependency": sentence.dependencyParse()
                })

            # Add the paragraph to our results
            results.append(par)

        return results, words