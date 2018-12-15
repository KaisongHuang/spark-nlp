#!/bin/bash

spark-2.4.0-bin-hadoop2.7/bin/spark-submit \
	--driver-memory 8G --master "local[*]" \
	--jars git/Anserini-Spark/target/anserini-spark-0.0.1-SNAPSHOT-fatjar.jar \
	src/main.py --index ~/lucene-index.wash18.pos+docvectors+rawdocs --num 100
