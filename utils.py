import glob
import os
import argparse

import datasets
import jsonlines
from tqdm import tqdm

batch_size = 4096
NUM_PROC = os.cpu_count()


def batch_iterator(
    dataset: datasets.arrow_dataset.Dataset,
    key: str = "text",
    batch_size: int = batch_size,
):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][key]


def text_iterator(text_path: str, key: str = "text", batch_size: int = batch_size):
    try:
        with open(
            text_path,
            "r",
            buffering=4096,
        ) as input_txt:
            while batch := input_txt.readlines(batch_size):
                yield batch
    except Exception as _:
        return None



def load_from_filepath(path: str, cache_dir: str, key="text"):
    dataset = None
    if os.path.isfile(path):
        if ".jsonl" in path:
            dataset = load_dataset(path)
        if ".txt" in path:
            dataset = load_text(path)
        if ".parquet" in path:
            dataset = datasets.Dataset.from_parquet(path, num_proc=NUM_PROC,cache_dir=cache_dir)
    return dataset


def load_from_path(path: str, cache_dir: str,key="text"):
    path = os.path.normpath(path).replace("~", os.path.expanduser("~"))
    if os.path.isfile(path):
        dataset = load_from_filepath(path,cache_dir=cache_dir)

    if os.path.isdir(path):
        try:
            # attempt to approach as single arrow Dataset dir
            dataset = datasets.load_from_disk(path)  # returns dataset
        except FileNotFoundError:
            ds_list = []
            globpath = os.path.join(path, "*")
            subpaths = glob.glob(globpath)
            for subpath in subpaths:
                if os.path.isfile(subpath):
                    ds = load_from_filepath(subpath)
                    if ds is not None:
                        ds_list.append(ds)
                if os.path.isdir(subpath):
                    try:
                        ds_list.append(datasets.load_from_disk(subpath))
                    except:
                        # don't go recursive.
                        pass
            if ds_list == []:
                dataset = None
            else:
                dataset = datasets.concatenate_datasets(ds_list)
    return dataset


def load_dataset(file_path, key="text"):
    data = []
    with jsonlines.open(file_path) as reader:
        for sample in reader.iter(allow_none=True, skip_empty=True, skip_invalid=True):
            try:
                data.append({key: sample[key]})
            except:
                pass
    dataset = datasets.Dataset.from_list(data)
    return dataset


def load_text(text_path: str, key: str = "text", num_proc: int = NUM_PROC):
    dataset = datasets.Dataset.from_text(text_path, num_proc)
    return dataset


def convert_json_to_dataset(file_path, save_path, key="text"):
    dataset = load_dataset(file_path, key)
    dataset.save_to_disk(save_path)


def filter_dataset(dataset: datasets.Dataset,key="text") -> datasets.Dataset:
    return dataset.filter(lambda example: example[key].strip(), num_proc=NUM_PROC)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="tokenizer model",
    )
    parser.add_argument("--dropout", default=None, help="dropout rate for BPE")
    parser.add_argument(
        "--vocab_size", type=int, default=50257, help="vocab size for tokenizer"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        # default="~/corpus/jsonl/the_stack_smol.jsonl",
        help="takes str to single file or path",
    )
    parser.add_argument(
        "--hf_ds_path",
        type=str,
        default="hac541309/open-lid-dataset",
        required=False,
        help="takes huggingface data repo",
    )
    parser.add_argument("--save_path", type=str, default="tokenizer")
    parser.add_argument(
        "--key",
        type=str,
        default="text",
        help="""key column to extract from dataset """,
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="tokenizer",
        help="""(output model prefix)  default:"tokenizer" """,
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default="NFC",
        choices=["NFKC", "NFC"],
        help="unicode normalizer NFKC is closer to spm. NFC is closer to most text",
    )
    parser.add_argument(
        "--buffer_tokens",
        type=int,
        default=64,
        help="number of tokens to pad BEFORE tokenizer initialization",
    )
    parser.add_argument(
        "--byte_fallback",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Huggingface style Bytelevel() vs spm style (byte_fallback)",
    )
    parser.add_argument(
        "--cache_capacity",
        type=int,
        default=10000,  # this is default for tokenizers. 
        help="cache_capacity in BPE.",
    )
    parser.add_argument(
        "--component_mode",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="unigram component mode",
    )
    parser.add_argument(
        "--whitespace_reservation",
        type=int,
        default=0,
        help="number of whitespaces to add as special tokens. \n \
            default length linear. \n \
            sorted down from len = (whitespace_reservation)",
        # consider no repeat ngrams during generation. (3 indentations--> bad)
    )
    parser.add_argument(
        "--exp_whitespace_reservation",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If this is set, whitespace changes into exponential mode \
            default length linear. \n \
            sorted down from len = (whitespace_reservation)",
        # consider no repeat ngrams during generation. (3 indentations--> bad)
    )
    parser.add_argument(
        "--preserve_whitespace",
        type=str,
        default="yes",
        choices=["yes", "inference", "no"],
        help="choose whitespace preservation. \n \
            yes: preserves during training and inference\n \
            inference: removes during training but resumes at inference\n \
            no: removes completely. this makes tokenizer non invertible(loses original)",
    )
    parser.add_argument(
        "--remove_longspace",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="""during training preprocessing, remove whitespaces longer than 16""",
    )
    parser.add_argument(
        "--single_whitespace",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to include single whitespace in vocab",
    )
    parser.add_argument(
        "--add_prefix_space",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="add prefix space. True : 'Gword','word' ",
    )
    parser.add_argument(
        "--isolate_camelcase",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="during tokenizer training preprocessing, isolate camelCases",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="corpus shuffle seed. 0(default) for no shuffling",
    )
    parser.add_argument(
        "--sample_percent",
        type=int,
        default=100,
        help="how much to sample",
    )
    parser.add_argument(
        "--max_sentence_length",
        type=int,
        default=4096,
        help="maximum length of sentence in characters",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=16,
        help="Prevents creating tokens longer than the specified size",
    )
    parser.add_argument(
        "--bpe_min_frequency",
        type=int,
        default=-1,
        help="(BPE)The minimum frequency a pair should have in order to be merged.",
    )

    args, _ = parser.parse_known_args()
    return args