from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import os

buffer_count = 76
buffer_tokens = [f"<|unused{i}|>" for i in range(buffer_count)]

tokenizer_path = os.path.join("05_02/base/tokenizer.json")
tokenizer_object = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer_object.get_vocab_size()

SPECIAL_TOKENS = [
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
    "<|sep|>",
    "<|mask|>",
]
added_tokens = SPECIAL_TOKENS + buffer_tokens
tokenizer_wrapper = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_object,
    vocab_size=vocab_size,
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    sep_token="<|sep|>",
    pad_token="<|pad|>",
    cls_token="<|cls|>",
    mask_token="<|mask|>",
    additional_special_tokens=added_tokens,
)
tokenizer_wrapper.save_pretrained("./wrap")
