#!/usr/bin/env bash

VENVNAME=glassenv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip --version
pip install --upgrade pip
pip --version

#sudo apt-get -y install graphviz graphviz-dev
#sudo apt-get -y install python3-graph-tool

pip install ipython
pip install jupyter
pip install matplotlib

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
