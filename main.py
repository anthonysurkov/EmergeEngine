from pathlib import Path
import pandas as pd

import emerge_mdl

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
INFILE = DATA_DIR / 'ctd_ad1_clean.csv'
#INFILE = DATA_DIR / 'ctd_joint_editors.csv'

BASES = ['A', 'G', 'C', 'U']
SEQ_COL = '5to3'

def main():
    # import corpus
    df = pd.read_csv(INFILE, index_col=0)
    df = df.rename(columns={'mle_ad1': 'mle', 'n_ad1': 'n', 'k_ad1': 'k'})
    df = df[df['n'] >= 10].sort_values(by='mle', ascending=False)
    df = df[df['mle'] >= 0.10]
    #df = df.head(1000)
    print(df)
    """
    df = pd.read_csv(INFILE, index_col=0)
    df = df[(df['n_ad1'] >= 6) & (df['n_ad2'] >= 6)]
    df = df.sort_values(by='mle_t', ascending=False)
    df_top = df.head(1000)
    print(df_top)
    n = df_top.shape[0]
    print(n)
    """

    emerge_forest = emerge_mdl.EmergeMDL(
        df_emerge = df,
        kmax = 6,
        seq_col = '5to3',
        n_col = 'n',
        k_col = 'k',
        to_rna = True
    )

    emerge_forest.encode(vocab_size=1000)
    emerge_forest.build_forest()

    list_before = emerge_forest.flatten()
    emerge_forest.prune_forest()
    list_after = emerge_forest.flatten()

    total_before = len(list_before)
    total_after = len(list_after)
    print(f'total before: {total_before}, total after: {total_after}')

    emerge_forest.append_edits()

    emerge_forest.to_html(outfile='test.html', min_length=0)

if __name__ == "__main__":
    main()
