from transformers import BertTokenizerFast
from transformers import BertConfig
import torch
import numpy as np
import random
from model import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_TokenizerConfig():
    tokenizer = BertTokenizerFast('../vocab.txt',do_lower_case=False)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=50,
        intermediate_size=1024,
        num_attention_heads=6,
        num_hidden_layers=4
    )
    return tokenizer, config


def load_model(config, use_property=False, CUDA=True):
    if CUDA:
        return MyBERT(config, use_property).cuda()
    else:
        return MyBERT(config, use_property)