from ..task import Task


class CoreNLPNamedEntityRecognition(Task):

    def __init__(self, spark_context):
        self.sc = spark_context

        # Get an instance of the Properties
        self.props = self.sc._jvm.java.util.Properties()
        self.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner")
        self.setProperty("ner.useSUTime", "false");

        # Get an instance of the Pipeline
        self.pipeline = self.sc._jvm.edu.stanford.nlp.pipeline.StanfordCoreNLP(self.props)

    def run(self, data):
        results = []
        words = 0

        # For each parsed doc...
        for paragraph in data:

            # Keep track of sentences in the paragraph
            par = []

            # For each sentence in the paragraph, fetch the entities
            anno = self.sc._jvm.edu.stanford.nlp.pipeline.Annotation(paragraph)
            self.pipeline.annotate(anno)
            sent_anno = self.sc._jvm.edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation()
            for sentence in anno.get(sent_anno.getClass()):
                words += len(sentence)

                # Get an instance of the CoreDocument
                doc = self.sc._jvm.edu.stanford.nlp.pipeline.CoreDocument(sentence)
                self.pipeline.annotate(doc)

                par.append({
                    "text": sentence,
                    "entities": self.get_entities(doc)
                })

            # Add the paragraph to our results
            results.append(par)

        return results, words

    def get_entities(self, sent):
        return [{em.text(): em.entityType()} for em in sent.entityMentions()]