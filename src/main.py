import argparse
import json
import time

import spacy
from pyspark import SparkContext
from pyspark.mllib.common import _java2py


def setup_spacy(gpu):
    if gpu:
        spacy.require_gpu()
    return spacy.load("en", disable=['parser', 'tagger'])


def get_docs(index):
    # Get an instance of the JavaIndexLoader
    index_loader = sc._jvm.io.anserini.spark.JavaIndexLoader(sc._jsc, index)

    # Get the document IDs as an RDD
    docids = index_loader.docIds()

    # Get an instance of our Lucene RDD class
    lucene = sc._jvm.io.anserini.spark.JavaLuceneRDD(docids)

    # Get the JavaRDD of Lucene Document as a Map (Document can't be serialized)
    docs = lucene.getDocs(index)

    # Convert to a Python RDD
    return _java2py(sc, docs)


# Get an array of paragraphs (str)
def get_paragraphs(raw):
    arr = []
    parsed = json.loads(raw)
    for content in parsed["contents"]:
        type = content["type"]
        if type == "sanitized_html":
            subtype = content["subtype"]
            if subtype == "paragraph":
                arr.append(content['content'])
    return arr


# Return a dict of entities to their labels
def extract_entites(text):
    ents = {}
    for ent in nlp(text).ents:
        ents[ent.text] = (ent.label_, ent.start_char, ent.end_char)
    return ents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=False, type=bool, help="whether to use GPUs for inference")
    parser.add_argument("--index", required=True, type=str, help="the index path")

    # Parse the args
    args = parser.parse_args()

    sc = SparkContext(master="local[*]")

    nlp = setup_spacy(args.gpu)
    docs = get_docs(args.index)

    start = time.time_ns()

    for doc in docs.take(100):
        for sent in get_paragraphs(doc["raw"]):
            entites = extract_entites(sent)
            if entites:
                print(extract_entites(sent))

    print("Took %d ms" % ((time.time_ns() - start) / 1000 / 1000))
