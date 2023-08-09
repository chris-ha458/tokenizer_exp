#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name="polyglot"
#SBATCH --mem=640GiB
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
cd _tokenizers/bindings/python &&\
python3 setup.py install && cd ../../..

# change this path
# xargs python3 ./train_tokenizer.py < /home/karyo/corpus/scripts/configs/unigram_test.txt

python3 train_tokenizer.py \
--model "bpe" \
--hf_ds_path "hac541309/open-lid-dataset" \
--key "response" \
--model_prefix "openLID" \
--whitespace_reservation 4 \
--exp_whitespace_reservation \
--sample_percent 100


deactivate

rm -rf _temp _tokenizers __temp__
