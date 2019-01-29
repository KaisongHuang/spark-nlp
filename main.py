import argparse
import json
import time

from pyspark import SparkContext

from libs.allen.ner import AllenNLPNamedEntityRecognition
from libs.allen.pos import AllenNLPPartOfSpeechTagger
from libs.allen.seg import AllenNLPSentenceSegmenter
from libs.corenlp.ner import CoreNLPNamedEntityRecognition
from libs.corenlp.pos import CoreNLPPartOfSpeechTagger
from libs.corenlp.seg import CoreNLPSentenceSegmenter
from libs.nltk.ner import NLTKNamedEntityRecognition
from libs.nltk.pos import NLTKPartOfSpeechTagger
from libs.nltk.seg import NLTKSentenceSegmenter
from libs.spacy.ner import SpacyNamedEntityRecognition
from libs.spacy.pos import SpacyPartOfSpeechTagger
from libs.spacy.seg import SpacySentenceSegmenter


# Get an array of paragraphs (str)
def get_paragraphs(document):
    paragraphs = []
    if (document is not None) and ("contents" in document):
        for content in document["contents"]:
            if (content is not None) and ("content" in content) and ("type" in content) and ("subtype" in content):
                if (content["type"] == "sanitized_html") and (content["subtype"] == "paragraph"):
                    paragraphs.append(content["content"])
    return paragraphs


def get_task():

    # AllenNLP
    if args.library == "allen":
        if args.task == "ner":
            return AllenNLPNamedEntityRecognition(args.gpu)
        if args.task == "pos":
            return AllenNLPPartOfSpeechTagger(args.gpu)
        if args.task == "seg":
            return AllenNLPSentenceSegmenter(args.gpu)

    # CoreNLP
    if args.library == "corenlp":
        if args.task == "ner":
            return CoreNLPNamedEntityRecognition({})
        if args.task == "pos":
            return CoreNLPPartOfSpeechTagger({})
        if args.task == "seg":
            return CoreNLPSentenceSegmenter({})

    # NLTK
    if args.library == "nltk":
        if args.task == "ner":
            return NLTKNamedEntityRecognition({})
        if args.task == "pos":
            return NLTKPartOfSpeechTagger({})
        if args.task == "seg":
            return NLTKSentenceSegmenter({})

    # spaCy
    if args.library == "spacy":
        if args.task == "ner":
            return SpacyNamedEntityRecognition(args.gpu)
        if args.task == "pos":
            return SpacyPartOfSpeechTagger(args.gpu)
        if args.task == "seg":
            return SpacySentenceSegmenter(args.gpu)


def process(part):
    results = []
    task = get_task()

    for doc in part:
        result, tokens = task.run(get_paragraphs(json.loads(doc)))
        results.append(result)
        total_tokens.add(tokens)

    return iter(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=str, help="the collection file")
    parser.add_argument("--library", default="spacy", type=str, help="allennlp vs. corenlp vs. nltk vs. spacy")
    parser.add_argument("--gpu", default=-1, type=int, help="the GPU number to use (spacy or allennlp)")
    parser.add_argument("--task", default="ner", type=str, help="ner vs. pos vs. seg")
    parser.add_argument("--sample", default=-1, type=float, help="the # of sample to take")

    # Parse the args
    args = parser.parse_args()

    # Create the SparkContext
    sc = SparkContext(appName="Spark NLP - {}:{}".format(args.library, args.task))

    # Keep track of the # of tokens processed for tokens / sec calculation
    total_tokens = sc.accumulator(0)

    start = time.time()

    # The collection file as a RDD
    rdd = sc.textFile(args.collection)

    if args.sample > 0:
        rdd.sample(False, args.sample).mapPartitions(process).foreach(lambda x: print(x))
    else:
        rdd.foreachPartition(process)

    total_time = time.time() - start

    print("Took %.2f s @ %.2f words/s" % (total_time, (total_tokens.value / total_time)))
