#emerge_language.py
from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
from collections import defaultdict, Counter, deque
from collections.abc import Iterable
from typing import Optional
import re
from tqdm.auto import tqdm
from emerge_data import EmergeHandler, MotifNode
from emerge_forest import MotifForest

class EmergeBPE(EmergeHandler):
    def __init__(
        self,
        df_emerge: pd.DataFrame,
        kmax: int = 6,
        special: Optional[set[str]] = None,
        seq_col: str = '5to3',
        n_col: str = 'n',
        k_col: str = 'k',
        to_rna: bool = True
    ) -> None:
        super().__init__(
            df_emerge = df_emerge,
            seq_col = seq_col,
            n_col = n_col,
            k_col = k_col,
            to_rna = to_rna
        )
        # check kmax
        if not isinstance(kmax, int):
            raise TypeError('arg `kmax` must be an integer greater than 1`')
        if kmax <= 2:
            raise ValueError('arg `kmax` must be an integer greater than 1`')
        if kmax > self.seq_len:
            raise ValueError('arg `kmax` must not exceed RNA length')

        if special is None:
            special = set()
        else:
            if not isinstance(special, set):
                raise TypeError(
                    'arg `special` must be a list of chars, '
                    f'got {type(special)}'
                )
            if not all(isinstance(x, str) and len(x) == 1 for x in special):
                raise ValueError(
                    'arg `special` must be a list of chars; '
                    f'at least one element is a str longer than 1 char'
                )
        special.add('E')

        # configure alphabet
        base_chars = set(''.join(self.df['5to3'].astype(str)))
        unexpected = base_chars - set(['A','G','C','U'])
        if unexpected:
            warnings.warn(
                'sequences contain chars outside canonical RNA nucleotides: '
                f'{unexpected}',
                UserWarning
            )
        base_chars = sorted(base_chars)
        alphabet = [
            f'{letter}{i}' for letter in base_chars
            for i in range(0, self.seq_len)
        ]
        tokens = alphabet + list(special)

        self.kmax = kmax
        self.vocab = {i: char for i, char in enumerate(tokens)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}
        self.base_chars = base_chars
        self.special = special

        self.merges = {}
        self.global_ranks = {}
        self.counts = Counter()

    def encode(self, vocab_size: int = 1000) -> None:
        if not isinstance(vocab_size, int) or vocab_size < 1:
            raise TypeError('arg `vocab_size` must be a positive integer')

        corpus = self._build_corpus()
        token_ids = self._corpus_to_token_ids(corpus)

        token_ids = self._learn_merges(token_ids, vocab_size)

        if not self.merges:
            warnings.warn('no merges made in encode()')

    def _build_corpus(self) -> list[str]:
        corpus: list[str] = []
        for seq in self.df['5to3']:
            for i, char in enumerate(seq):
                corpus.append(f'{char}{i}')
            corpus.append('E')
        return corpus

    def _corpus_to_token_ids(self, corpus: list[str]) -> list[int]:
        return [self.inverse_vocab[token] for token in corpus]

    def _learn_merges(
        self,
        token_ids: list[int],
        vocab_size: int
    ) -> list[int]:
        global_rank = 0

        # for tqdm progress bar
        start = len(self.vocab)
        target = vocab_size
        total_merges = max(0, target - start)
        pbar = tqdm(total=total_merges, desc='BPE merges', unit='tok')

        for new_id in range(start, target):
            pair_id = self._find_freq_pair(
                token_ids=token_ids,
                vocab=self.vocab,
                mode='most'
            )
            if pair_id is None:
                pbar.update(target - pbar.n)
                break
            p0, p1 = pair_id
            merged_token = self.vocab[p0] + self.vocab[p1]

            token_ids = self._replace_pair(token_ids, pair_id, new_id)
            global_rank += 1
            self.merges[pair_id] = new_id
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
            self.global_ranks[(p0, p1)] = global_rank

            k = sum(ch in self.base_chars for ch in merged_token)

            pbar.update(1)

        pbar.close()
        return token_ids

    def to_forest(self) -> MotifForest:
        nodes: dict[int, MotifNode] = {}

        def make_node(node_id: int) -> MotifNode:
            if node_id not in nodes:
                nodes[node_id] = MotifNode(
                    node_id = node_id,
                    motif_seq = self.vocab[node_id]
                )
            return nodes[node_id]

        base_pat = re.compile(r'[ACGU]\d+$')
        for tid, tok in self.vocab.items():
            if base_pat.fullmatch(tok):
                nd = make_node(tid)
                nd.motif_seq = tok
                nd.seqs = self.get_motif_seqs(tok)
                nd.prevalence = len(nd.seqs)

        total = len(self.merges.items())
        pbar = tqdm(total=total, desc='Forest gen', unit='node')
        for (left_id, right_id), child_id in self.merges.items():
            child = make_node(child_id)
            left = make_node(left_id)
            right = make_node(right_id)

            child.add_parent(left)
            child.add_parent(right)

            left.motif_seq = self.vocab[left_id]
            right.motif_seq = self.vocab[right_id]
            child.motif_seq = self.vocab[child_id]

            left.seqs = self.get_motif_seqs(left.motif_seq)
            right.seqs = self.get_motif_seqs(right.motif_seq)
            child.seqs = self.get_motif_seqs(child.motif_seq)

            child.prevalence = len(child.seqs)
            left.prevalence = len(left.seqs)
            right.prevalence = len(right.seqs)

            pbar.update(1)

        pbar.close()

        roots = [nd for nd in nodes.values() if len(nd.parents) == 0]
        return MotifForest(roots, self.df)

    def _find_freq_pair(
        self,
        token_ids: list[int],
        vocab: dict[int, str],
        mode: str = 'most'
    ) -> Optional[tuple[int, int]]:
        if self.base_chars is None:
            self.base_chars = ['A','C','G','U']

        pairs = Counter(zip(token_ids, token_ids[1:]))

        filtered: dict[tuple[int, int], int] = {}
        for (p0, p1), count in pairs.items():
            if vocab is not None:
                tok0 = vocab[p0]
                tok1 = vocab[p1]

                if (
                    any(ch in self.special for ch in tok0) or
                    any(ch in self.special for ch in tok1)
                ):
                    continue

                merged = tok0 + tok1
                merged_len = sum(ch in self.base_chars for ch in merged)
                if merged_len > self.kmax:
                    continue

            filtered[(p0, p1)] = count

        if not filtered:
            return None

        if mode == 'most':
            return max(filtered.items(), key = lambda x: x[1])[0]
        elif mode == 'least':
            return min(filtered.items(), key = lambda x: x[1])[0]
        else:
            raise ValueError('invalid mode (`most` or `least`)')

    @staticmethod
    def _replace_pair(
        token_ids: list[int],
        pair_id: tuple[int, int],
        new_id: int
    ) -> list[int]:
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

