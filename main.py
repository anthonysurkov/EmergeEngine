from pathlib import Path
import pandas as pd

import emerge_tokenizer as et
import emerge_vis

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
INFILE = DATA_DIR / 'ctd_ad1_map.csv'

BASES = ['A', 'G', 'C', 'U']
SEQ_COL = '5to3'

def main():
    df = pd.read_csv(INFILE, index_col=0)
    df = df.rename(columns={'...1.y': 'map', 'n10': '5to3'})
    df = df[df['n'] >= 6].sort_values(by='map', ascending=False)
    df_top = df.head(1000)
    print(df_top)

    tokenizer1 = et.EmergeTokenizer(
        df_emerge = df_top,
        kmax = 6,
        seq_col = '5to3',
        edit_col = 'map',
        to_rna = True
    )
    tokenizer1.encode()
    emerge_forest = tokenizer1.build_forest()

    title = 'CTD1 AD2 Motif Forest'
    subtitle = """
        0-indexed. NNNNNNNNNN = 0123456789<br>
        e.g. G6A7U8 is NNNNNNGAUN<br>
        e.g. U0U1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;is UUNNNNNNNN<br><br>
        Mouse over each node for more information.<br>
        Rank refers to the order in which byte-pair encoding (BPE) identified
         the motifs, adjusted for k-mer size. For example, rank 1 means that
         it was the first k-mer of its length to be identified, making it the
         most frequent motif of its length.<br>
        Count is the number of times the motif appears in the high-editing
         population. The size of a node also roughly corresponds to
         its prevalence.<br>
        Average editing is taken across the entire population of sequences,
         not just the high-editing population.
    """
    outfile = 'ctd_ad2_forest.html'
    emerge_vis.forest_to_html(
        emerge_forest = emerge_forest,
        outfile = outfile,
        title = title,
        subtitle = subtitle
    )

if __name__ == '__main__':
    main()
