from pathlib import Path
import pandas as pd

from emerge_data import EmergePredicate, EmergeHandler
from emerge_language import EmergeBPE
from emerge_forest import MotifForest
from emerge_phenotype import ForestPhenotyper

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
INFILE = DATA_DIR / 'r270x_z.csv'

def main():
    df = pd.read_csv(INFILE, index_col=0)
    df = df.rename(columns={'n_ad1': 'n', 'k_ad1': 'k'})

    emerge_handle = EmergeHandler(df_emerge = df)

    edited_10 = EmergePredicate.col_geq('mle', 0.50)
    n_10 = EmergePredicate.col_geq('n', 10)
    df = emerge_handle.query(edited_10, n_10, columns=['5to3','n','k','mle'])
    print(df)

    special = set('Z')
    emerge_bpe = EmergeBPE(
        df_emerge=df,
        kmax=6,
        special=special
    )
    emerge_bpe.encode()
    emerge_forest = emerge_bpe.to_forest().with_canopy()

    emerge_phenotyper = ForestPhenotyper(forest=emerge_forest)

    list_before = emerge_forest.flatten()
    emerge_phenotyper.prune_by_enrichment()
    list_after = emerge_forest.flatten()

    total_before = len(list_before)
    total_after = len(list_after)
    print(f'total before: {total_before}, total after: {total_after}')

    emerge_forest.traverse(func=emerge_phenotyper.append_edits_bb)

    emerge_forest.to_html(outfile='test.html', min_length=0)

if __name__ == "__main__":
    main()
