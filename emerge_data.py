#emerge_data.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any
import functools
import operator
import warnings

import pandas as pd
from pandas.api.types import is_bool_dtype
import numbers
import re

class EmergePredicate:
    def __init__(
        self,
        func: Callable[[pd.DataFrame], pd.Series],
        name: Optional[str] = None
    ) -> None:
        self._func = func
        self.name = name or getattr(func, "__name__", "predicate")

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        mask = self._func(df)

        if not isinstance(mask, pd.Series):
            raise TypeError(
                f'predicate {self.name!r} must return a pandas Series, '
                f'got {type(mask)}'
            )
        if mask is None:
            raise ValueError(
                f'boolean mask is empty; cannot be'
            )
        if not is_bool_dtype(mask.dtype):
            raise TypeError(
                f'predicate {self.name!r} must return a boolean Series, '
                f'got dtype {mask.dtype}'
            )
        if not mask.index.equals(df.index):
            raise ValueError(
                f'predicate {self.name!r} returned a Series with a different '
                f'index than the input DataFrame'
            )

        return mask

    def __and__(self, other: EmergePredicate) -> EmergePredicate:
        if not isinstance(other, EmergePredicate):
            return NotImplemented

        def _and(df: pd.DataFrame) -> pd.Series:
            return self(df) & other(df)

        return EmergePredicate(_and, name=f'({self} & {other})')

    def __or__(self, other: EmergePredicate) -> EmergePredicate:
        if not isinstance(other, EmergePredicate):
            return NotImplemented

        def _or(df: pd.DataFrame) -> pd.Series:
            return self(df) | other(df)

        return EmergePredicate(_or, name=f'({self} | {other})')

    def __invert__(self) -> EmergePredicate:
        def _not(df: pd.DataFrame) -> pd.Series:
            return ~self(df)

        return EmergePredicate(_not, name=f'(~{self})')

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'EmergePredicate(name={self.name!r})'

    @staticmethod
    def col_geq(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] >= value
        return EmergePredicate(_pred, name=f'{col} >= {value}')

    @staticmethod
    def col_leq(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] <= value
        return EmergePredicate(_pred, name=f'{col} <= {value}')

    @staticmethod
    def col_gt(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] > value
        return EmergePredicate(_pred, name=f'{col} > {value}')

    @staticmethod
    def col_lt(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] < value
        return EmergePredicate(_pred, name=f'{col} < {value}')

    @staticmethod
    def col_eq(col:str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] == value
        return EmergePredicate(_pred, name=f'{col} == {value}')

    @staticmethod
    def str_contains(
        col: str,
        pattern: str,
        regex: bool = True
    ) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col].astype('string').str.contains(pattern, regex=regex)
        return EmergePredicate(_pred, name=f'{col}.contains({pattern!r})')

class EmergeHandler:
    def __init__(
        self,
        df_emerge: pd.DataFrame,
        seq_col: str = '5to3',
        n_col: str = 'n',
        k_col: str = 'k',
        to_rna: bool = True
    ):
        if not isinstance(df_emerge, pd.DataFrame):
            raise TypeError(
                'arg `df_emerge` must be a pandas DataFrame. '
                f'is: {type(df_emerge)}'
            )
        df_emerge = df_emerge.copy(deep=True)

        if not isinstance(seq_col, str):
            raise TypeError(
                'arg `seq_col` must be a string'
                f'is: {type(seq_col)}'
            )
        if not isinstance(n_col, str):
            raise TypeError(
                'arg `n_col` must be a string'
                f'is: {type(n_col)}'
            )
        if not isinstance(k_col, str):
            raise TypeError(
                'arg `k_col` must be a string'
                f'is: {type(k_col)}'
            )
        if df_emerge.shape[0] == 0:
            raise ValueError(
                '`df_emerge` must have more than one entry associated with it'
            )
        elif seq_col not in df_emerge.columns:
            raise KeyError('arg `seq_col` is not a valid col in `df_emerge`')
        elif k_col not in df_emerge.columns:
            raise KeyError('arg `k_col` is not a valid col in `df_emerge`')
        elif n_col not in df_emerge.columns:
            raise KeyError('arg `n_col` is not a valid col in `df_emerge`')
        if df_emerge[[seq_col, n_col, k_col]].isnull().any().any():
            raise ValueError(
                '`df_emerge` cannot contain NaN values` in neither `seq_col`'
                ' nor `n_col` nor `k_col`.'
            )
        if not all(isinstance(x, str) for x in df_emerge[seq_col]):
            raise TypeError('Every element of `seq_col` must be a string')
        if not all(isinstance(x, numbers.Integral) for x in df_emerge[k_col]):
            raise TypeError('Every element of `k_col` must be an int')
        if not all(isinstance(x, numbers.Integral) for x in df_emerge[n_col]):
            raise TypeError('Every element of `n_col` must be an int')

        seq_len = len(df_emerge[seq_col].iloc[0])
        if df_emerge[seq_col].str.len().nunique() != 1:
            raise ValueError(
                'Every element of `seq_col` must have the same length'
            )
        if seq_len == 0:
            raise ValueError('sequences cannot be empty (seq_len == 0)')

        df_emerge[seq_col] = df_emerge[seq_col].str.upper()
        if to_rna:
            df_emerge[seq_col] = (
                df_emerge[seq_col].str.replace('T','U', regex=False)
            )

        cols = [seq_col, n_col, k_col]
        self.df = df_emerge[cols]
        self.df = self.df.rename(
            columns={seq_col: '5to3', n_col: 'n', k_col: 'k'}
        )
        self.df['mle'] = self.df['k'] / self.df['n']
        self.seq_len = seq_len
        self.df_len = self.df.shape[0]

    def query(
        self,
        *predicates: EmergePredicate,
        columns: Optional[Iterable[str]] = None,
        copy: bool = True
    ) -> pd.DataFrame:
        df = self.df

        combined = self._combine_predicates(predicates)
        if combined is not None:
            mask = combined(df)
            df = df[mask]

        if columns is None:
            columns = ['5to3','n','k','mle']
        df = df.loc[:, list(columns)]

        return df.copy(deep=True) if copy else df

    def get_motif_seqs(
        self,
        motif: str
    ) -> pd.DataFrame:
        request = self.contains_motif(motif)
        mask = request(self.df)
        return self.query(request)

    # TODO: provide motif spec option that isn't A0C1G2 etc.
    # want something like ACTG0-3. also support for gappy motifs, e.g.
    # ACTGXXXGT0-8. serves as a nice internal identifier for motifs too
    # specify with bpe_syntax=False in args
    def contains_motif(
        self,
        motif: str
    ) -> EmergePredicate:
        mapping = self._parse_token(motif)
        mask = self._token_to_mask(mapping)

        def _match(seq) -> bool:
            #if seq is None or pd.isna(seq):
            #    return False
            #seq = str(seq)
            return self._matches_mask(seq, mask)

        def _pred(df: pd.DataFrame) -> pd.Series:
            return df['5to3'].apply(_match)

        return EmergePredicate(_pred, name=f'5to3.contains({motif!r})')

    def _token_to_mask(self, mapping: dict[int, str]) -> str:
        mask = ["X"] * self.seq_len
        for pos, base in mapping.items():
            mask[pos] = base
        return "".join(mask)

    def _combine_predicates(
        self,
        predicates: Sequence[EmergePredicate]
    ) -> Optional[EmergePredicate]:
        if not predicates:
            return None
        return functools.reduce(operator.and_, predicates)

    @staticmethod
    def _parse_token(seq: str) -> dict[int, str]:
        token_re = re.compile(r'([ACGU])(\d+)')
        return {int(pos): base for base, pos in token_re.findall(seq)}

    @staticmethod
    def _matches_mask(seq: str, mask: str) -> bool:
        return all(m == "X" or s == m for s, m in zip(seq, mask))

class MotifEdge:
    def __init__(
        self,
        parent: MotifNode,
        child: MotifNode,
        delta: Optional[float] = None,
        pval: Optional[float] = None,
        sp: bool = False
    ):
        self.parent = parent
        self.child = child
        self.delta = delta
        self.pval = pval
        self.sp = sp

        self.parent_id = id(self.parent)
        self.child_id = id(self.child)

# eq=False
class MotifNode:
    def __init__(
        self,
        motif_seq: str = None,
        node_id: int = None,
        seqs: Optional[pd.DataFrame] = None,
        rank: Optional[int] = None,
        prevalence: Optional[int] = None,
        edit: Optional[float] = None,
        special_chars: Optional[list[str]] = None
    ):
        self.motif_seq = motif_seq
        self.node_id = node_id
        self.seqs = seqs
        self.rank = rank
        self.prevalence = prevalence
        self.edit = edit

        self.motif_state = None
        self.pval = None
        self.sp = None

        self.parents = []
        self.children = []

        sc = set(special_chars) if special_chars else {'Z'}

        if motif_seq is None:
            self.base_chars = []
            self.motif_len = 0
        else:
            self.base_chars = sorted({
                ch for ch in self.motif_seq
                if (not ch.isdigit()) and (ch not in sc)
            })
            self.motif_len = sum((ch in self.base_chars) for ch in motif_seq)

    def __repr__(self):
        J = 0 if self.seqs is None else len(self.seqs)
        return (
            f'MotifNode(id={self.node_id}, motif={self.motif_seq}, J={J}, '
            f'parents={len(self.parents)}, children={len(self.children)})'
        )

    def add_child(self, child: MotifNode) -> None:
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def add_parent(self, parent: MotifNode) -> None:
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)

