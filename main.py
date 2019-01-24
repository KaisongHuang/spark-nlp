import argparse
import json
import time

from pyspark import SparkContext


def get_docs(index):
    from pyspark.mllib.common import _java2py

    # Get an instance of the JavaIndexLoader
    index_loader = sc._jvm.io.anserini.spark.IndexLoader(sc._jsc, index)

    # Get the document IDs as an RDD
    docids = index_loader.docids()

    # Get the JavaRDD of Lucene Document as a Map (Document can't be serialized)
    docs = index_loader.docs2map(docids, index)

    # Convert to a Python RDD
    return _java2py(sc, docs)


# Get an array of paragraphs (str)
def get_paragraphs(document):
    arr = []
    if (document is not None) and ("contents" in document):
        for content in document["contents"]:
            if (content is not None) and ("content" in content) and ("type" in content) and ("subtype" in content):
                if (content["type"] == "sanitized_html") and (content["subtype"] == "paragraph"):
                    arr.append(content["content"])
    return arr


def run(doc):
    paragraphs = get_paragraphs(json.loads(doc["raw"]))
    result, words = task.run(paragraphs)
    total_words.add(words)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=str, help="the index path")
    parser.add_argument("--library", default="spacy", type=str, help="spacy vs. stanford vs. nltk")
    parser.add_argument("--task", default="ner", type=str, help="ner vs. pos vs. seg")
    parser.add_argument("--num", default=-1, type=int, help="the number of documents use")

    # Parse the args
    args = parser.parse_args()

    sc = SparkContext(appName="Spark NLP - {}:{}".format(args.library, args.task))

    # Keep track of the # of words processed for words / sec calculation
    total_words = sc.accumulator(0)

    # Get the RDD of Lucene Documents
    docs = get_docs(args.index)

    if args.library == "spacy":
        if args.task == "ner":
            from libs.spacy.ner import SpacyNamedEntityRecognition
            task = SpacyNamedEntityRecognition({})

    start = time.time()

    if args.num < 1:
        docs.foreach(lambda doc: run(doc))
    else:
        for doc in docs.take(args.num):
            print("###\n# Document ID: %s\n###" % doc["id"])
            for paragraph in run(doc):
                print(paragraph)

    total_time = time.time() - start

    print("Took %.2f s @ %.2f words/s" % (total_time, (total_words.value / total_time)))
