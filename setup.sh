#!/bin/bash

echo "Helllooooooooooooooooooooo worlddddddddddd"
mkdir -p /opt/render/project/src/nltk_data
export NLTK_DATA=/opt/render/project/src/nltk_data
python -m nltk.downloader all

