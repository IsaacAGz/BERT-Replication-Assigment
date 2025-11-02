

#Used to detect whitespace, or accents
import unicodedata

#Used to support Unicode property classes
import regex as re

#Huggingface transformers BertTokenizer
from transformers import BertTokenizer

#For Dictionary
import collections

#downloads vocabulary
tok = BertTokenizer.from_pretrained("bert-base-uncased")

#vocabulary size
print(tok.vocab_size) 

NEVER_SPLIT = {"[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"}

UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "[PAD]"
MASK = "[MASK]"

#regex pattern for unicode punctuation
_PUNC_RE = re.compile(r"([\p{P}])")

#Cleans the text

#determines if character is a whitespace
def _is_whitespace(ch):
    return ch in (" ", "\t", "\n", "\r") or unicodedata.category(ch) == "Zs"

#strips accents and returns string without accents
def _strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

#checks to see if character is a control or whitespace character
def _is_control(ch):
    cat = unicodedata.category(ch)
    return (cat.startswith("C") and ch not in ("\t", "\n", "\r"))

#rebuilds the text sting by removing control characters, null characters, or whitespaces, then returns it.
def _clean_text(text):
    out = []
    for ch in text:
        if ch == "\u0000" or _is_control(ch):
            continue
        out.append(" " if _is_whitespace(ch) else ch)
    return "".join(out)
    
#returns an ordered vocab dictionary
def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as f:
        for i, token in enumerate(f):
            token = token.rstrip("\n")
            vocab[token] = i
    return vocab
    
#Tokenizes usisng wodpiece tokenizer vocabulary
class WordpieceTokenizer:

    def __init__(self, vocab, unk_token = UNK, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self,token):
        
        if len(token) > self.max_input_chars_per_word:
            return [self.unk_token]

        sub_tokens = []
        start = 0
        while start < len(token):
            end = len(token)
            cur_substring = None

            while start < end:
                substring = token[start:end]

                if start > 0:
                    substring = "##" + substring

                if substring in self.vocab:
                    cur_substring = substring
                    break

                end -= 1

            if cur_substring is None:
                return [self.unk_token]

            sub_tokens.append(cur_substring)

            start = end
            
        return sub_tokens
class FullTokenizer:

    def __init__(self, vocab_file, do_lower_case = True, never_split = None):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v:k for k, v in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.never_split = set(NEVER_SPLIT if never_split is None else never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(self.vocab, unk_token= UNK)
        

    def tokenize(self, text):
        if not text:
            return []

        text = _clean_text(text)

        #make lower case and remove accents
        if self.do_lower_case:
            text = text.lower()
            text = _strip_accents(text)

        #split on whitespace and punctuation, keeping punctuation as token
        tokens = []
        for tok in text.strip().split():
            if tok in self.never_split:
                tokens.append(tok)
                continue
            parts = [p for p in _PUNC_RE.split(tok) if p and not p.isspace()]
            tokens.extend(parts)

        #wordpiece token list
        wp_tokens = []
        for t in tokens:
            if t in self.never_split:
                wp_tokens.append(t)
            else:
                wp_tokens.extend(self.wordpiece_tokenizer.tokenize(t))

        return wp_tokens

    def convert_tokens_to_ids(self, tokens):
        unk_id = self.vocab.get(UNK)

        return [self.vocab.get(t, unk_id) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[i] for i in ids]

def build_inputs_from_tokens(tokens_a, tokens_b = None, max_len = 512, pad_to_max = True, pad_token = PAD):
    tokens = [CLS] + tokens_a + [SEP]

    token_type_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + [SEP]
        token_type_ids += [1] * (len(tokens_b) + 1)

    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        token_type_ids = token_type_ids[:max_len]

    attention_mask = [1] * len(tokens)

    if pad_to_max and len(tokens) < max_len:
        pad_len = max_len - len(tokens)
        tokens += [pad_token] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    return tokens, token_type_ids, attention_mask

def build_inputs_from_texts(tokenizer, text_a, text_b = None, max_len = 512):
    ta = tokenizer.tokenize(text_a)
    tb = tokenizer.tokenize(text_b) if text_b is not None else None
    tokens, token_type_ids, attention_mask = build_inputs_from_tokens(ta, tb, max_len=max_len)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return dict(
        input_ids = input_ids,
        token_type_ids = token_type_ids,
        attention_mask = attention_mask,
        tokens = tokens
    )