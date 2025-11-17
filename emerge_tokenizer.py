from __future__ import annotations
from dataclasses import dataclass, field
import warnings
from numbers import Real

import pandas as pd
from collections import Counter, deque, defaultdict
import re

# Helper functions (not public; not type-protected)
def find_freq_pair(
    token_ids: list[str],
    vocab: dict[int, str],
    mode: str = 'most'
) -> tuple[int, int]:
    pairs = Counter(zip(token_ids, token_ids[1:]))
    if vocab is not None:
        pairs = {
            p: c for p, c in pairs.items()
            if 'E' not in vocab[p[0]] and 'E' not in vocab[p[1]]
        }
    if mode == 'most':
        return max(pairs.items(), key = lambda x: x[1])[0]
    elif mode == 'least':
        return min(pairs.items(), key = lambda x: x[1])[0]
    else:
        raise ValueError('invalid mode (`most` or `least`)')

def replace_pair(
    token_ids: list[int],
    pair_id: tuple[int, int],
    new_id: int
):
    dq = deque(token_ids)
    replaced = []
    while dq:
        current = dq.popleft()
        if dq and (current, dq[0]) == pair_id:
            replaced.append(new_id)
            dq.popleft()
        else:
            replaced.append(current)
    return replaced

def count_kmers(
    seqs: list[str],
    kmax: int
):
    counts = Counter()
    for seq in seqs:
        seq = seq.replace('T','U')
        tokens = [f"{char}{i}" for i, char in enumerate(seq)]
        for k in range(2, kmax + 1):
            for i in range(len(tokens) - k + 1):
                kmer = "".join(tokens[i:i+k])
                counts[kmer] += 1
    return counts

token_re = re.compile(r'([ACGU])(\d+)')
def parse_token(seq: str) -> dict[int, str]:
    return {int(pos): base for base, pos in token_re.findall(seq)}

def token_to_mask(mapping: dict[int, str], L: int) -> str:
    mask = ["X"] * L
    for pos, base in mapping.items():
        mask[pos] = base
    return "".join(mask)

def matches_mask(seq: str, mask: str) -> bool:
    return all(m == "X" or s == m for s, m in zip(seq, mask))

def find_matching_seqs(
    node: MotifNode,
    df: pd.DataFrame,
    seq_col: str = 'seq'
):
    token = node.motif_seq
    mapping = parse_token(token)
    L = len(df[seq_col].iloc[0])
    mask = token_to_mask(mapping, L)
    def match(s):
        s = s.replace('T', 'U')
        return matches_mask(s, mask)
    sub = df[df[seq_col].apply(match)]
    return sub[seq_col]

def find_avg_edit(
    node: MotifNode,
    df: pd.DataFrame,
    seq_col: str = 'seq',
    edit_col: str = 'edit'
):
    token = node.motif_seq
    mapping = parse_token(token)
    L = len(df[seq_col].iloc[0])
    mask = token_to_mask(mapping, L)
    def match(s):
        s = s.replace('T', 'U')
        return matches_mask(s, mask)
    sub = df[df[seq_col].apply(match)]
    return sub[edit_col].mean()

def append_avg_edits_to_tree(
    node: MotifNode,
    df: pd.DataFrame,
    seq_col='seq',
    edit_col='edit'
):
    avg_edit = find_avg_edit(node, df, seq_col, edit_col)
    node.editing = avg_edit
    for child in node.children:
        append_avg_edits_to_tree(child, df, seq_col, edit_col)

# public
@dataclass
class MotifNode:
    def __init__(
        self,
        node_id: MotifNode | None = None,
        motif_seq: str | None = None,
        seqs: pd.DataFrame = None,
        rank: int | None = None,
        prevalence: int | None = None,
        editing: float | None = None,
        parents: list[MotifNode] | None = None,
    ):
        self.node_id = node_id
        self.motif_seq = motif_seq,
        self.seqs = seqs
        self.rank = rank
        self.prevalence = prevalence
        self.editing = editing
        self.parents = parents or []
        self.children = []

    def add_child(self, child: MotifNode):
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)
        return self

    def add_parent(self, parent: MotifNode):
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)
        return self

class EmergeTokenizer:
    def __init__(
        self,
        df_emerge: pd.DataFrame,
        kmax: int = 6,
        seq_col: str = '5to3',
        edit_col: str = 'mle',
        to_rna: bool = True
    ):
        df_emerge = df_emerge.copy(deep=True)

        if not isinstance(df_emerge, pd.DataFrame):
            raise TypeError(
                'arg `df_emerge` must be a pandas DataFrame. '
                f'is: {type(df_emerge)}'
            )
        if not isinstance(seq_col, str):
            raise TypeError(
                'arg `seq_col` must be a string'
                f'is: {type(seq_col)}'
            )
        if not isinstance(edit_col, str):
            raise TypeError(
                'arg `edit_col` must be a string'
                f'is: {type(edit_col)}'
            )
        if df_emerge.shape[0] == 0:
            raise ValueError(
                '`df_emerge` must have more than one entry associated with it'
            )
        elif seq_col not in df_emerge.columns:
            raise KeyError('arg `seq_col` is not a valid col in `df_emerge`')
        elif edit_col not in df_emerge.columns:
            raise KeyError('arg `edit_col` is not a valid col in `df_emerge`')
        if df_emerge[[seq_col, edit_col]].isnull().any().any():
            raise ValueError(
                '`df_emerge` cannot contain NaN values` in neither `edit_col`'
                ' nor `seq_col`.'
            )
        if not all(isinstance(x, str) for x in df_emerge[seq_col]):
            raise TypeError('Every element of `seq_col` must be a string')
        if not all(isinstance(x, Real) for x in df_emerge[edit_col]):
            raise TypeError('Every element of `edit_col` must be numbers.Real')

        # configure seq_len
        seq_len = len(df_emerge[seq_col].iloc[0])
        if df_emerge[seq_col].str.len().nunique() != 1:
            raise ValueError(
                'Every element of `seq_col` must have the same length'
            )
        if seq_len == 0:
            raise ValueError('sequences cannot be empty (seq_len == 0)')

        # convert to RNA sequences
        df_emerge[seq_col] = df_emerge[seq_col].str.upper()
        if to_rna:
            df_emerge[seq_col] = (
                df_emerge[seq_col].str.replace('T','U', regex=False)
            )

        # configure alphabet
        alphabet = set(''.join(df_emerge[seq_col].astype(str)))
        unexpected = alphabet - set(['A','G','C','U'])
        if unexpected:
            warnings.warn(
                'sequences contain chars outside canonical RNA nucleotides: '
                f'{unexpected}',
                UserWarning
            )
        alphabet = [
            f'{letter}{i}' for letter in alphabet for i in range(0, seq_len)
        ]

        # check kmax
        if not isinstance(kmax, int):
            raise TypeError('arg `kmax` must be an integer greater than 1`')
        if kmax <= 2:
            raise ValueError('arg `kmax` must be an integer greater than 1`')
        if kmax > seq_len:
            raise ValueError('arg `kmax` must not exceed RNA length')

        cols = [seq_col, edit_col]
        self.df = df_emerge[cols]
        self.df = self.df.rename(
            columns={seq_col: 'seq', edit_col: 'edit'}
        )
        self.kmax = kmax
        self.seq_len = seq_len
        self.vocab = {i: char for i, char in enumerate(alphabet)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        self.merges = {}
        self.global_ranks = {}
        self.kmer_ranks = {}
        self.kmer_bins = defaultdict(list)
        self.counts = Counter()

    def encode(
        self,
        vocab_size: int = 1000,
        special: set[str] | None = None
    ):
        if special is None:
            special = {'E'}
        if not isinstance(special, set):
            raise TypeError('arg `special` must be set of single-char strings')
        for x in special:
            if not isinstance(x, str) or len(x) != 1:
                raise ValueError('special token {x!r} must be a single char')
        special.add('E')

        if not isinstance(vocab_size, int):
            raise TypeError('arg `vocab_size` must be a positive integer')
        if vocab_size < 1:
            raise TypeError('arg `vocab_size` must be a positive integer')

        corpus = []
        for seq in self.df['seq']:
            for i, char in enumerate(seq):
                corpus.append(f'{char}{i}')
            corpus.append('E')

        if special:
            for token in special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        token_ids = [self.inverse_vocab[token] for token in corpus]

        global_rank = 0
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = find_freq_pair(token_ids, self.vocab, 'most')
            if pair_id is None:
                break
            p0, p1 = pair_id
            merged_token = self.vocab[p0] + self.vocab[p1]
            if len(merged_token) > 2 * self.kmax:
                continue

            token_ids = replace_pair(token_ids, pair_id, new_id)
            global_rank += 1
            self.merges[pair_id] = new_id
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
            self.global_ranks[(p0, p1)] = global_rank

            k = len(merged_token)
            self.kmer_bins[k].append(new_id)

        for k, ids in self.kmer_bins.items():
            for i, tid in enumerate(ids, start=1):
                self.kmer_ranks[tid] = i

        self.counts = count_kmers(self.df['seq'], self.kmax)

    def build_forest(self):
        nodes = {}
        def get_node(node_id):
            if node_id not in nodes:
                nodes[node_id] = MotifNode(
                    node_id = node_id,
                    motif_seq = str(node_id)
                )
            return nodes[node_id]

        for (left_id, right_id), parent_id in self.merges.items():
            parent = get_node(parent_id)
            left = get_node(left_id)
            right = get_node(right_id)

            parent.add_child(left)
            parent.add_child(right)

            left.motif_seq = self.vocab[left_id]
            right.motif_seq = self.vocab[right_id]
            parent.motif_seq = self.vocab[parent_id]

            left.seqs = find_matching_seqs(left, self.df, seq_col = 'seq')
            right.seqs = find_matching_seqs(right, self.df, seq_col = 'seq')
            parent.seqs = find_matching_seqs(parent, self.df, seq_col = 'seq')

            parent.rank = self.kmer_ranks.get(parent_id)
            parent.prevalence = len(parent.seqs)
            left.prevalence = len(left.seqs)
            right.prevalence = len(right.seqs)

        all_children = {c for (l, r) in self.merges for c in (l, r)}
        roots = [
            get_node(pid) for (l, r), pid in self.merges.items()
            if pid not in all_children
        ]
        for root in roots:
            append_avg_edits_to_tree(
                root, self.df, seq_col='seq', edit_col='edit'
            )
        return roots
