import json
import logging
import os
from time import time

import datasets
from tokenizers import Regex, Tokenizer, decoders, normalizers
from tokenizers.models import BPE, Unigram
from tokenizers.pre_tokenizers import (
    ByteLevel,
    Digits,
    Punctuation,
    Sequence,
    Split,
    UnicodeScripts,
)
from tokenizers.trainers import BpeTrainer, UnigramTrainer

from utils import (
    batch_iterator,
    filter_dataset,
)

logger = logging.getLogger()
NUM_PROC = os.cpu_count()
CACHE_DIR = "./__temp__/tokenizer_corpus"
HF_PATH = "hac541309/open-lid-dataset"
DS_PATH = "/home/karyo/corpus/data/txt_small"
KEY = "text"
VOCAB_SIZE = 160_000
BUFFER_TOKEN_COUNT = 64
MODEL = ["bpe", "unigram"][0]
MODEL_PREFIX = "_".join([MODEL, str(VOCAB_SIZE // 1000), "k"])
SAVE_PATH = "./model/"


def main():
    training_dataset = datasets.load_dataset(
        path=DS_PATH,num_proc=NUM_PROC, cache_dir=CACHE_DIR,split='train',
    )
    num_examples = len(training_dataset)
    training_dataset = filter_dataset(training_dataset, key=KEY)
    print(num_examples)

    # tokenizer arguments
    SPECIAL_TOKENS = sorted(
        [
            "<|sep|>",
            "<|s|>",
            "<|/s|>",
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|endoftext|>",
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|translate|>",
            "<|startofprompt|>",
            "<|endofprompt|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<|meta|>",
            "<|meta_start|>",
            "<|meta_end|>",
            "<|mask|>",
            "<|mask1|>",
            "<|cls|>",
            "<|cls_vision|>",
            "<|cls_audio|>",
            "<|tel_start|>",
            "<|tel_end|>",
            "<|rrn_start|>",
            "<|rrn_end|>",
            "<|url_start|>",
            "<|url_end|>",
            "<|email_start|>",
            "<|email_end|>",
            "<|crd_start|>",
            "<|crd_end|>",
            "<|acc_start|>",
            "<|acc_end|>",
            "<|name_start|>",
            "<|name_end|>",
            "<|org_start|>",
            "<|org_end|>",
            "<|sos|>",
            "<|unk|>",
        ]
    )
    # calculate and construct whitespace tokens
    whitespace = " "

    whitespace_list = [whitespace * n for n in [2, 4, 8, 16]]

    vocab_size = VOCAB_SIZE - len(whitespace_list)  # we will add whitespace later

    # construct buffer_tokens
    buffer_tokens = [f"<|unused{i:02d}|>" for i in range(BUFFER_TOKEN_COUNT)]
    # construct added_token
    added_tokens = sorted((SPECIAL_TOKENS + buffer_tokens))
    trainer_parameters = {
        "vocab_size": vocab_size,
        "special_tokens": added_tokens,
    }
    # normalizer and common pretokenizer
    normalizer = normalizers.NFC()
    pre_tokenizer_list = [
        UnicodeScripts(),  # split on different unicode range
        Punctuation(
            behavior="isolated",  # isolated vs contiguous /* */  /*******/
        ),
        Digits(individual_digits=True),
    ]

    # camelCase logic
    camel_case_pattern = Regex(r"(?<=[a-z])(?=[A-Z])")
    pre_tokenizer_list.append(
        Split(
            pattern=camel_case_pattern,
            behavior="isolated",
            invert=False,
        )
    )
    pre_tokenizer_list.append(ByteLevel(add_prefix_space=False, use_regex=True))
    decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer_parameters["initial_alphabet"] = sorted(ByteLevel.alphabet())

    # construct_pretokenizer
    pre_tokenizer = Sequence(pre_tokenizer_list)

    if MODEL == "bpe":
        tokenizer = Tokenizer(
            BPE(
                cache_capacity=VOCAB_SIZE,
                unk_token="<|unk|>",
            )
        )

        trainer_parameters["max_token_length"] = 16
        # trainer_parameters["min_frequency"] = 100
        trainer = BpeTrainer(**trainer_parameters)
    elif MODEL == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer_parameters["max_piece_length"] = 16
        trainer = UnigramTrainer(**trainer_parameters)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder

    start = time()
    tokenizer.train_from_iterator(
            batch_iterator(training_dataset, key=KEY),
            trainer=trainer,
            length=num_examples,
        )

    end = time()
    print(f"time elapsed: {(end - start) / 60:.2f}m")

    # if preserve whitespace is set to "inference", remove Whitespace splitter

    # remove preprocess only pretokenizers.
    for item in pre_tokenizer_list:
        if isinstance(item, Split):
            pre_tokenizer_list.remove(item)
            break
    tokenizer.pre_tokenizer = Sequence(pre_tokenizer_list)
    tokenizer.add_tokens(whitespace_list)

    # need to add as special then unspecialize
    # assert tokenizer.add_tokens(fallback_hexadecimals) == 256

    # save tokenizer
    save_path = f"{SAVE_PATH}/{MODEL_PREFIX}.json"
    os.makedirs(SAVE_PATH, exist_ok=True)
    tokenizer.save(path=save_path, pretty=True)
    tokenizer = Tokenizer.from_file(save_path)

    text = "아!@ ㈝12시 진짜홀수ws     tOkeN  짝수ws    억울해죽것네'''newLine\nNewline taB\tTab 아니enGlish123배고파씌 Korea으😣악😣😳😣'''"

    # transformers
    # tokens = tokenizer_wrapper.tokenize(text)
    # no transformers
    tokens = tokenizer.encode(text).tokens
    # transformers
    # input_ids = tokenizer_wrapper(text)["input_ids"]
    input_ids = tokenizer.encode(text).ids
    vocab_dict_vk = {v: k for k, v in tokenizer.get_vocab().items()}

    decoded_dict = {}
    for idx, token in vocab_dict_vk.items():
        decoded_dict[idx] = tokenizer.decoder.decode([token])
    each_decode = []
    ufb_count = 0
    for id in input_ids:
        decoded = decoded_dict[id]
        if decoded == "�":
            each_decode.append("|ufb|")
            ufb_count += 1
        else:
            each_decode.append(decoded)
    decoded = tokenizer.decode(input_ids)

    print("{} {}".format(MODEL, VOCAB_SIZE))
    print(f"text:{text}")

    LEN_TOKENS = len(tokens)
    print(f"token count: {LEN_TOKENS}")
    print(
        f"unicode fallback portion: \
            {ufb_count} /{LEN_TOKENS} = {ufb_count / LEN_TOKENS:.3f}"
    )
    if MODEL == "unigram":
        sum = 0
        with open(save_path, "r") as tokenizer_json:
            model_json = json.load(tokenizer_json)
            for token in model_json["model"]["vocab"]:
                sum += token[1]
            print(f"unigram ALP : {sum / vocab_size}")


if __name__ == "__main__":
    main()
