#!/bin/bash
wget https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz

# untar to folder
tar -xvzf tvsum50_ver_1_1.tgz
rm -rf tvsum50_ver_1_1.tgz

# converts space separated file names to underscore separated names
rm -rf videos
unzip ydata-tvsum50-v1_1/ydata-tvsum50-video.zip -d videos

rm -rf GT
unzip ydata-tvsum50-v1_1/ydata-tvsum50-data.zip -d GT