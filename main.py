import argparse
import json
import time
import os
import shutil

from pyspark import SparkContext
from libs.allen.ner import AllenNLPNamedEntityRecognition
from libs.allen.pos import AllenNLPPartOfSpeechTagger
from libs.allen.dep import AllenNLPDependencyParsing
from libs.stanfordnlp.pos import StanfordNLPPartOfSpeechTagger
from libs.stanfordnlp.dep import StanfordNLPDependencyParsing
from libs.nltk.ner import NLTKNamedEntityRecognition
from libs.nltk.pos import NLTKPartOfSpeechTagger
from libs.spacy.dep import SpacyDependencyParser
from libs.spacy.ner import SpacyNamedEntityRecognition
from libs.spacy.pos import SpacyPartOfSpeechTagger
from htmltextparser import HTMLTextParser


# Get the document's id
def get_docid(document):
    docid = ''
    if (document is not None) and ("id" in document):
        docid = document["id"]
    return docid


# Get an array of paragraphs (str)
def get_paragraphs(document):
    paragraphs = []
    if (document is not None) and ("contents" in document):
        for content in document["contents"]:
            if (content is not None) and ("content" in content) and ("type" in content) and ("subtype" in content):
                if (content["type"] == "sanitized_html") and (content["subtype"] == "paragraph"):
                    parser = HTMLTextParser()
                    parser.feed(str(content["content"]))
                    paragraph = parser.get_text()
                    paragraphs.append(paragraph)
    return paragraphs


def get_task():
    # AllenNLP
    if args.library == "allennlp":
        if args.task == "ner":
            return AllenNLPNamedEntityRecognition(args.gpu)
        if args.task == "pos":
            return AllenNLPPartOfSpeechTagger(args.gpu)
        if args.task == "dep":
            return AllenNLPDependencyParsing(args.gpu)

    # NLTK
    if args.library == "nltk":
        if args.task == "ner":
            return NLTKNamedEntityRecognition({})
        if args.task == "pos":
            return NLTKPartOfSpeechTagger({})

    # spaCy
    if args.library == "spacy":
        if args.task == "dep":
            return SpacyDependencyParser(args.gpu)
        if args.task == "ner":
            return SpacyNamedEntityRecognition(args.gpu)
        if args.task == "pos":
            return SpacyPartOfSpeechTagger(args.gpu)


def process(part):
    results = []

    if args.library != "stanfordnlp":
        task = get_task()
    else:
        task = _task

    for doc in part:
        docid = get_docid(json.loads(doc))
        if args.library != "stanfordnlp":
            result, tokens = task.run(get_paragraphs(json.loads(doc)))
        else:
            result, tokens = task.value.run(get_paragraphs(json.loads(doc)))
        results.append((docid, result))
        total_tokens.add(tokens)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=str, help="the collection file")
    parser.add_argument("--library", default="spacy", type=str, help="allennlp vs. stanfordnlp vs. nltk vs. spacy")
    parser.add_argument("--gpu", default=-1, type=int, help="the GPU number to use (spacy or allennlp)")
    parser.add_argument("--task", default="ner", type=str, help="ner vs. pos vs. seg")
    parser.add_argument("--sample", default=-1, type=float, help="the % of sample to take")
    parser.add_argument("--output", required=True, type=str, help="the path to the output files")

    # Parse the args
    args = parser.parse_args()

    # Delete output directory if exists
    if os.path.isdir(args.output):
        shutil.rmtree(args.output)

    # Create the SparkContext
    sc = SparkContext(appName="Spark NLP - {}:{}".format(args.library, args.task))

    # Keep track of the # of tokens processed for tokens / sec calculation
    total_tokens = sc.accumulator(0)

    start = time.time()

    # The collection file as a RDD
    rdd = sc.textFile(args.collection)

    # Initialize a StanfordNLP task and broadcast it
    if args.library == "stanfordnlp":
        if args.task == "pos":
            task = StanfordNLPPartOfSpeechTagger(args.gpu)
        if args.task == "dep":
            task = StanfordNLPDependencyParsing(args.gpu)
        _task = sc.broadcast(task)

    if args.sample > 0:
        # result_list = rdd.sample(False, args.sample).mapPartitions(process).collect()# foreach(lambda x: print(x))
        # sc.parallelize(result_list).saveAsTextFile(args.output)
        rdd.sample(False, args.sample).mapPartitions(process).saveAsTextFile(args.output)
    else:
        rdd.foreachPartition(process)

    total_time = time.time() - start

    print("Took %.2f s @ %.2f words/s" % (total_time, (total_tokens.value / total_time)))
