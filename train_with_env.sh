#!/bin/bash

#SBATCH --requeue
#SBATCH --job-name="polyglot"
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --partition=cpu128
#SBATCH --mem=972GB
#SBATCH --time=24:00:00
#SBATCH --error=/fsx/polyglot-tokenizer/logs/slurm-%j.err
#SBATCH --output=/fsx/polyglot-tokenizer/logs/slurm-%j.out
#SBATCH --comment=eleuther


#setup env
source ~/dev_envs/env_tokenizers/bin/activate


# download and install huggingface tokenizers
# git clone https://github.com/huggingface/tokenizers.git tokenizers_env &&\
# cd tokenizers_env/bindings/python && python setup.py install
# cd ../../..

# change this path
# xargs python3 ~/scripts/train_tokenizer.py < ~/scripts/configs/polyglot_64k.txt

python3 train_tokenizer.py \
--model "bpe" \
--data_path "/home/karyo/corpus/data/openlid/parquet/openlid.parquet" \
--key "text" \
--model_prefix "openLID" \
--whitespace_reservation 4 \
--max_sentence_length 0 \
--exp_whitespace_reservation \
--sample_percent 100


deactivate

#rm -rf .cache __cache__ __*
