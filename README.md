Anserini-Spark
-

At the moment, [`Anserini-Spark`](https://github.com/castorini/Anserini-Spark) needs to be built with version `0.3.1-SNAPSHOT` of [`Anserini`](https://github.com/castorini/Anserini). You will need to build `Anserini` such that it is available in your local `maven` repo and then build `Anserini-Spark`.

Setup
-

The following script will setup the virtualenv + download Anserini-Spark and any other required software:

`./setup.sh`

Running
-

We're ready to run now (after changing the parameters in the script for index location, etc.):

`./run.sh`