#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name="polyglot"
#SBATCH --mem=320GB
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --error=/fsx/polyglot-tokenizer/logs/slurm-%j.err
#SBATCH --output=/fsx/polyglot-tokenizer/logs/slurm-%j.out
#SBATCH --comment=eleuther


# silent rust setup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

#setup env
python3 -m venv _temp && source _temp/bin/activate
pip install setuptools_rust datasets jsonlines


# download and install huggingface tokenizers
git clone https://github.com/huggingface/tokenizers.git _tokenizers &&\
cd _tokenizers/bindings/python && python setup.py install
cd ../../..

# change this path
# xargs python3 ./train_tokenizer.py < /home/karyo/corpus/scripts/configs/unigram_test.txt

deactivate

rm -rf _temp _tokenizers ~/corpus/__temp__