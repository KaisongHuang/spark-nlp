import argparse
import json
import time

from pyspark import SparkContext


def get_docs(index):
    from pyspark.mllib.common import _java2py

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


def run_spacy(doc):
    paragraphs = get_paragraphs(json.loads(doc["raw"]))
    return spacy_ner.ner(nlp, paragraphs)


def run_allen(doc):
    paragraphs = get_paragraphs(json.loads(doc["raw"]))
    return allen_ner.ner(nlp, predictor, paragraphs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=-1, type=int, help="the GPU number to use")
    parser.add_argument("--index", required=True, type=str, help="the index path")
    parser.add_argument("--num", default=-1, type=int, help="the number of documents use")
    parser.add_argument("--library", default="spacy", type=str, help="spacy vs. allennlp")

    # Parse the args
    args = parser.parse_args()

    sc = SparkContext(appName="spaCy NER")

    # Get the RDD of Lucene Documents
    docs = get_docs(args.index)

    start = time.time()

    if args.library == "spacy":

        import spacy_ner

        # Setup spacy
        nlp = spacy_ner.setup(args.gpu)

        if args.num < 1:
            docs.foreach(lambda doc: run_spacy(doc))
        else:
            for doc in docs.take(args.num):
                print("###\n# Document ID: %s\n###" % doc["id"])
                run_spacy(doc)
                # for paragraph in run(doc):
                # print(paragraph)

    if args.library == "allennlp":

        import allen_ner

        nlp, predictor = allen_ner.setup()

        if args.num < 1:
            docs.foreach(lambda doc: run_allen(doc))

    print("Took %d ms" % (time.time() - start))
