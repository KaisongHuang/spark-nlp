import argparse
import json
import time

import spacy_ner
from pyspark import SparkContext
from pyspark.mllib.common import _java2py


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
    return spacy_ner.ner(nlp, paragraphs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=-1, type=int, help="the GPU number to use")
    parser.add_argument("--index", required=True, type=str, help="the index path")
    parser.add_argument("--num", default=-1, type=int, help="the number of documents use")

    # Parse the args
    args = parser.parse_args()

    sc = SparkContext(appName="spacy NER")

    # Get the RDD of Lucene Documents
    docs = get_docs(args.index)

    start = time.time_ns()

    # Setup spacy
    nlp = spacy_ner.setup(args.gpu)

    if args.num < 1:
        docs.foreach(lambda doc: run(doc))
    else:
        for doc in docs.take(args.num):
            print("###\n# Document ID: %s\n###" % doc["id"])
            run(doc)
            # for paragraph in run(doc):
                # print(paragraph)

    print("Took %d ms" % ((time.time_ns() - start) / 1000 / 1000))    
