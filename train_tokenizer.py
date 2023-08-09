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
    Whitespace,
)
from tokenizers.trainers import BpeTrainer, UnigramTrainer

from utils import (
    batch_iterator,
    filter_dataset,
    load_from_path,
    parse_args,
)

logger = logging.getLogger()
NUM_PROC = os.cpu_count()
CACHE_DIR = "./__temp__/tokenizer_corpus"


def main(args):
    if args.data_path:
        training_dataset = load_from_path(args.data_path,cache_dir=CACHE_DIR)
    elif args.hf_ds_path:
        training_dataset = datasets.load_dataset(
            path=args.hf_ds_path,
            cache_dir=CACHE_DIR,
            num_proc=NUM_PROC,
            split="train",
        )
    else:
        raise ValueError("Check --data_path or hf_ds_path")
    num_examples = len(training_dataset)
    if args.sample_percent != 100:
        num_to_keep = int(num_examples * (args.sample_percent / 100))
        training_dataset = training_dataset.select(range(num_to_keep))
        if args.shuffle_seed == 0:
            training_dataset.shuffle(seed=42)
    if args.shuffle_seed != 0:
        training_dataset.shuffle(args.shuffle_seed)
    if args.max_sentence_length > 0:
        training_dataset = training_dataset.filter(
            lambda example: len(example[args.key]) <= args.max_sentence_length,
            num_proc=NUM_PROC,
        )
    training_dataset = filter_dataset(training_dataset, args.key)
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
    whitespace_count = args.whitespace_reservation  # 4,2 whitespaces

    if args.exp_whitespace_reservation:
        whitespace_list = (
            [whitespace * count for count in range(whitespace_count, 1, -1)]
            if whitespace_count > 0
            else []
        )
    else:
        whitespace_list = [
            whitespace * (2**count) for count in range(whitespace_count, 0, -1)
        ]

    if args.single_whitespace:
        whitespace_list.append(" ")  # add single_whitespace
    vocab_size = args.vocab_size - len(whitespace_list)  # we will add whitespace later

    # construct buffer_tokens
    buffer_token_count = args.buffer_tokens
    buffer_tokens = [f"<|unused{i:02d}|>" for i in range(buffer_token_count)]
    # construct added_token
    added_tokens = sorted(
        (
            SPECIAL_TOKENS
            + buffer_tokens
            # + whitespace_list #not a special token
        )
        if not args.component_mode
        else []
    )
    _ = args.add_prefix_space

    # tokenizer normalizer
    if args.normalizer.lower() == "nfc":
        normalizer = normalizers.NFC()
    elif args.normalizer.lower() == "nfkc":
        normalizer = normalizers.NFKC()

    # use Split() to prevent long spaces. allow up to (17 - 1) whitespace tokens
    # also splits camel case
    split_regex = r"s{17,}"
    split_pattern = Regex(split_regex)

    # common pretokenizer
    pre_tokenizer_list = [
        UnicodeScripts(),  # split on different unicode range
        Punctuation(
            behavior="isolated",  # isolated vs contiguous /* */  /*******/
        ),
        Digits(individual_digits=True),
    ]
    if args.remove_longspace:
        pre_tokenizer_list.append(
            Split(pattern=split_pattern, behavior="removed", invert=False)
        )
    # camel case logic
    if args.isolate_camelcase:
        camel_case_regex = r"(?<=[a-z])(?=[A-Z])"
        camel_case_pattern = Regex(camel_case_regex)
        pre_tokenizer_list.append(
            Split(
                pattern=camel_case_pattern,
                behavior="isolated",
                invert=False,
            )
        )
    trainer_parameters = {
        "vocab_size": vocab_size,
        "special_tokens": added_tokens,
    }

    # set byte_fallback and change pretokenizer and decoder
    byte_fallback = args.byte_fallback
    if byte_fallback:
        decoder = decoders.ByteFallback()
        first_ws_filter = Regex(r"(\b\w+\b|\s+\w+)")
        pre_tokenizer_list.append(Split(first_ws_filter, "isolated"))
        # build fallback characters and make room for them in vocab_size
        fallback_hexadecimals = [f"<0x{num:02X}>" for num in range(256)]
        fallback_tuples = [(fallback, 0.0) for fallback in fallback_hexadecimals]
        trainer_parameters["special_tokens"].extend(fallback_hexadecimals)
        trainer_parameters["vocab_size"] -= 256
    else:
        pre_tokenizer_list.append(ByteLevel(add_prefix_space=False, use_regex=True))
        decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
        if not args.component_mode:
            trainer_parameters["initial_alphabet"] = sorted(ByteLevel.alphabet())

    # if yes, default to no whitespace handling
    if not args.preserve_whitespace == "yes":
        pre_tokenizer_list.insert(
            0,
            Whitespace(),  # WhitespaceSplit()
        )  # whitespace split should be in front
    # common decoder

    # construct_pretokenizer
    pre_tokenizer = Sequence(pre_tokenizer_list)

    if args.model == "bpe":
        assert not args.component_mode
        tokenizer = Tokenizer(
            BPE(
                cache_capacity=args.cache_capacity,
                dropout=args.dropout,
                byte_fallback=byte_fallback,
                unk_token="<|unk|>",
            )
        )

        if args.max_token_length > 0:
            trainer_parameters["max_token_length"] = args.max_token_length
        if args.bpe_min_frequency > 0:
            trainer_parameters["min_frequency"] = args.bpe_min_frequency
        trainer = BpeTrainer(**trainer_parameters)
    elif args.model.lower() == "unigram":
        tokenizer = Tokenizer(Unigram())
        if not args.component_mode:
            trainer_parameters["unk_token"] = "<|unk|>"

        if args.max_token_length > 0:
            trainer_parameters["max_piece_length"] = args.max_token_length
        trainer = UnigramTrainer(**trainer_parameters)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder

    start = time()
    if isinstance(training_dataset, datasets.arrow_dataset.Dataset):
        tokenizer.train_from_iterator(
            batch_iterator(training_dataset, key=args.key),
            trainer=trainer,
            length=num_examples,
        )

    end = time()
    print(f"time elapsed: {(end - start) / 60:.2f}m")

    # if preserve whitespace is set to "inference", remove Whitespace splitter
    if args.preserve_whitespace == "inference":
        for item in pre_tokenizer_list:
            if isinstance(item, Whitespace):
                pre_tokenizer_list.remove(item)
                break

    # remove preprocess only pretokenizers.
    if args.remove_longspace:
        for item in pre_tokenizer_list:
            if isinstance(item, Split):
                pre_tokenizer_list.remove(item)
                break
    if args.isolate_camelcase:
        for item in pre_tokenizer_list:
            if isinstance(item, Split):
                pre_tokenizer_list.remove(item)
                break
    tokenizer.pre_tokenizer = Sequence(pre_tokenizer_list)
    tokenizer.add_tokens(whitespace_list)

    # need to add as special then unspecialize
    # assert tokenizer.add_tokens(fallback_hexadecimals) == 256

    # save tokenizer
    # tokenizer_wrapper.save_pretrained(f"{args.save_path}")
    model_name = f"{args.model_prefix}_{args.vocab_size // 10**3}k"
    save_path = f"{args.save_path}/{model_name}.json"
    os.makedirs(args.save_path, exist_ok=True)
    tokenizer.save(path=save_path, pretty=True)
    if byte_fallback:
        with open(save_path, "r") as json_input:
            tokenizer_json = json.load(json_input)
        tokenizer_json["model"]["byte_fallback"] = True
        # unspecialize
        for token in tokenizer_json["added_tokens"]:
            if token["content"] in fallback_hexadecimals:
                token["special"] = False
        with open(save_path, "w") as json_output:
            json.dump(tokenizer_json, json_output, indent=2, ensure_ascii=False)
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

    # wrap tokenizer
    # tokenizer_wrapper = PreTrainedTokenizerFast(        tokenizer_object=tokenizer,        vocab_size=args.vocab_size,        additional_special_tokens=added_tokens,        bos_token=SPECIAL_TOKENS[0],         eos_token=SPECIAL_TOKENS[0],          unk_token=SPECIAL_TOKENS[0],      )
    # tokens = tokenizer_wrapper.tokenize(text)
    # input_ids = tokenizer_wrapper(text)["input_ids"]
    # vocab_dict_vk = {v: k for k, v in tokenizer_wrapper.vocab.items()}
    # tokenizer_wrapper.save_pretrained(f"{args.save_path}")

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

    print("{} {} {}".format(args.model, args.vocab_size, args.sample_percent))
    print(f"text:{text}")

    if byte_fallback:
        recon = tokenizer.decode(tokenizer.encode(text).ids)
        print(f"decode:{recon}")
        print(f"invertible: {recon==text}")
    else:
        print(f"decode:{(decoded)}")
        print(f"invertible: {decoded==text}")
    print(f"tokens: {tokens}")
    if not byte_fallback:
        print(f"tokens recon: {each_decode}")
    print(
        f"original input length: char= {len(text)}, bytes= {len(text.encode('utf-8'))}"
    )
    LEN_TOKENS = len(tokens)
    print(f"token count: {LEN_TOKENS}")
    print(
        f"unicode fallback portion: {ufb_count} /{LEN_TOKENS} = {ufb_count / LEN_TOKENS:.3f}"
    )
    if args.model.lower() == "unigram":
        sum = 0
        with open(save_path, "r") as tokenizer_json:
            model_json = json.load(tokenizer_json)
            for token in model_json["model"]["vocab"]:
                sum += token[1]
            print(f"unigram ALP : {sum / vocab_size}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
