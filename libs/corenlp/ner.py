from ..task import Task


class CoreNLPNamedEntityRecognition(Task):

    def __init__(self, spark_context):
        self.sc = spark_context

        # Get an instance of the Properties
        self.props = self.sc._jvm.java.util.Properties()
        self.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner")
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

            # For each sentence in the paragraph, fetch the entities
            for sentence in doc.sentences():
                words += len(sentence)

                par.append({
                    "text": sentence,
                    "entities": self.get_entities(sentence)
                })

            # Add the paragraph to our results
            results.append(par)

        return results, words

    def get_entities(self, sent):
        return [{em.text(): em.entityType()} for em in sent.entityMentions()]
