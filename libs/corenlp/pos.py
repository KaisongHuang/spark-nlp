from ..task import Task


class CoreNLPPartOfSpeechTagger(Task):

    def __init__(self, spark_context):
        self.sc = spark_context

        # Get an instance of the Properties
        self.props = self.sc._jvm.java.util.Properties()
        self.setProperty("annotators", "tokenize,ssplit")
        self.setProperty("ner.useSUTime", "false");

        # Get an instance of the Pipeline
        self.pipeline = self.sc._jvm.edu.stanford.nlp.pipeline.StanfordCoreNLP(self.props)

    def run(self, data):
        paragraphs = []
        words = 0

        for paragraph in data:
            sentences = []
            anno = self.sc._jvm.edu.stanford.nlp.pipeline.Annotation(paragraph)
            self.pipeline.annotate(anno)
            sent_anno = self.sc._jvm.edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation()

            for sentence in anno.get(sent_anno.getClass()):
                doc = self.sc._jvm.edu.stanford.nlp.pipeline.CoreDocument(sentence)
                self.pipeline.annotate(doc)
                tokens = doc.tokens()

                for token in tokens:
                    pos_anno = self.sc._jvm.edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation()
                    sentences.append(" ".join("{}[{}]".format(token, token.get(pos_anno.getClass()))))
                words += len(tokens)

        return paragraphs, words
