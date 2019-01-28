import argparse
import json
import time

from pyspark import SparkContext
from pyspark.mllib.common import _java2py

from libs.corenlp.ner import CoreNLPNamedEntityRecognition
from libs.corenlp.pos import CoreNLPPartOfSpeechTagger
from libs.corenlp.seg import CoreNLPSentenceSegmenter
from libs.nltk.ner import NLTKNamedEntityRecognition
from libs.nltk.pos import NLTKPartOfSpeechTagger
from libs.nltk.seg import NLTKSentenceSegmenter
from libs.spacy.ner import SpacyNamedEntityRecognition
from libs.spacy.pos import SpacyPartOfSpeechTagger
from libs.spacy.seg import SpacySentenceSegmenter


def get_docs(index):
    # Get an instance of the IndexLoader
    index_loader = sc._jvm.io.anserini.spark.IndexLoader(sc._jsc, index)

    # Get the document IDs as an RDD
    docids = index_loader.docids()

    # Get the JavaRDD of Lucene Documents as a Map (Document can't be serialized)
    docs = index_loader.docs2map(docids, index)

    # Convert to a PythonRDD
    return _java2py(sc, docs)


# Get an array of paragraphs (str)
def get_paragraphs(document):
    paragraphs = []
    if (document is not None) and ("contents" in document):
        for content in document["contents"]:
            if (content is not None) and ("content" in content) and ("type" in content) and ("subtype" in content):
                if (content["type"] == "sanitized_html") and (content["subtype"] == "paragraph"):
                    paragraphs.append(content["content"])
    return paragraphs


def run(doc):
    paragraphs = get_paragraphs(json.loads(doc["raw"]))
    result, words = task.run(paragraphs)
    total_words.add(words)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=str, help="the index path")
    parser.add_argument("--library", default="spacy", type=str, help="corenlp vs. nltk vs. spacy")
    parser.add_argument("--task", default="ner", type=str, help="ner vs. pos vs. seg")
    parser.add_argument("--num", default=-1, type=int, help="the number of documents use")

    # Parse the args
    args = parser.parse_args()

    sc = SparkContext(appName="Spark NLP - {}:{}".format(args.library, args.task))

    # Keep track of the # of words processed for words / sec calculation
    total_words = sc.accumulator(0)

    # Get the RDD of Lucene Documents
    docs = get_docs(args.index)

    # CoreNLP
    if args.library == "corenlp":
        if args.task == "ner":
            task = CoreNLPNamedEntityRecognition(sc)
        if args.task == "pos":
            task = CoreNLPPartOfSpeechTagger(sc)
        if args.task == "seg":
            task = CoreNLPSentenceSegmenter(sc)

    # NLTK
    if args.library == "nltk":
        if args.task == "ner":
            task = NLTKNamedEntityRecognition({})
        if args.task == "pos":
            task = NLTKPartOfSpeechTagger({})
        if args.task == "seg":
            task = NLTKSentenceSegmenter({})

    # spaCy
    if args.library == "spacy":
        if args.task == "ner":
            task = SpacyNamedEntityRecognition({})
        if args.task == "pos":
            task = SpacyPartOfSpeechTagger({})
        if args.task == "seg":
            task = SpacySentenceSegmenter({})

    start = time.time()

    if args.num < 1:
        docs.foreach(lambda doc: run(doc))
    else:
        for doc in docs.take(args.num):
            print("\n###\n# Document ID: %s\n###\n" % doc["id"])
            i = 0
            for paragraph in run(doc):
                print("# Paragraph {}:".format(i))
                print(paragraph)
                print()
                i += 1

    total_time = time.time() - start

    print("Took %.2f s @ %.2f words/s" % (total_time, (total_words.value / total_time)))
