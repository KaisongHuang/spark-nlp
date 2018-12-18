#!/bin/bash

DIR=$1

for file in $DIR/*; do
    cat $file | ssh raclancy@datasci.cs.uwaterloo.ca "/usr/lib/hadoop/bin/hadoop fs -put - /user/raclancy/$file"
done
