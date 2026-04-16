from pathlib import Path
import pandas as pd

from emerge_data import EmergePredicate, EmergeHandler
from emerge_language import EmergeBPE
from emerge_forest import MotifForest
from emerge_phenotype import ForestPruner

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
INFILE = DATA_DIR / 'proqr.csv'
#INFILE = DATA_DIR / 'r255x_top.csv'

def main():
    wc = False

    #df = pd.read_csv(INFILE)
    df = pd.read_csv(INFILE, index_col=0)
    #df = df.rename(columns={'n_ad1': 'n', 'k_ad1': 'k'})
    #print(df)

    print(df)

    emerge_handle = EmergeHandler(df_emerge = df, edit_col = 'proqr_edit')

    #edited_10 = EmergePredicate.col_geq('mle', 0.30)
    #n_10 = EmergePredicate.col_geq('n', 10)
    #df = emerge_handle.query(edited_10, n_10, columns=['5to3','n','k','mle'])
    #print(df)

    special = set('Z')
    emerge_bpe = EmergeBPE(
        df_emerge=df,
        edit_col='proqr_edit',
        kmax=6,
        special=special
    )
    emerge_bpe.encode()
    emerge_forest = emerge_bpe.to_forest()
    #emerge_pruner = ForestPruner(emerge_forest, with_canopy=False)
    #emerge_pruner.kill_by_delta(with_canopy=wc, with_parents=False)

    df = emerge_forest.to_pd(with_canopy=wc)#
    print(df)
    #df_alive = df[df['motif_state'] > 0]
    #print(df_alive)
    #df_motif = df[df['motif_state'] == 2]

    emerge_forest.to_html(
        outfile='test_b.html',
        with_canopy=False,
        color_by='motif_status'
    )

if __name__ == "__main__":
    main()
