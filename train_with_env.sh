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
xargs python3 /home/karyo/corpus/scripts/train_tokenizer.py < /home/karyo/corpus/scripts/configs/polyglot_64k.txt
deactivate

#rm -rf .cache __cache__ __*
