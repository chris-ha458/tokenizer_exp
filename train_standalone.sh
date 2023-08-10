#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name="polyglot"
#SBATCH --mem=960GB
#SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --error=/admin/home-chris458/tokenizer_exp/logs/slurm-%j.err
#SBATCH --output=/admin/home-chris458/tokenizer_exp/logs/slurm-%j.out
#SBATCH --comment=eleuther


# silent rust setup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

#setup env
python3 -m venv _temp && source _temp/bin/activate
pip3 install setuptools_rust datasets jsonlines


# download and install huggingface tokenizers
rm -rf _tokenizers
git clone https://github.com/huggingface/tokenizers.git _tokenizers &&\
cd _tokenizers/bindings/python &&\
python3 setup.py install && cd ../../..
rm -rf _tokenizers

# change this path
# xargs python3 ./train_tokenizer.py < /home/karyo/corpus/scripts/configs/unigram_test.txt

python3 train_tokenizer_standalone.py

deactivate

rm -rf _temp _tokenizers __temp__
