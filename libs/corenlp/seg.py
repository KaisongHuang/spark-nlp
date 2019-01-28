from ..task import Task


class CoreNLPSentenceSegmenter(Task):

    def __init__(self, spark_context):
        self.sc = spark_context

        # Get an instance of the Properties
        self.props = self.sc._jvm.java.util.Properties()
        self.setProperty("annotators", "tokenize")
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
                sentences.append(str(sentence))
                words += len(sentence)
            paragraphs.append(sentences)
        return paragraphs, words