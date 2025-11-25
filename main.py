from pathlib import Path
import pandas as pd

from emerge_data import EmergeHandler
from emerge_language import EmergeBPE
from emerge_phenotype import ForestPhenotype

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
INFILE = DATA_DIR / 'ctd_ad1_clean.csv'

def main():
    df = pd.read_csv(INFILE, index_col=0)
    df = df.rename(columns={'mle_ad1': 'mle', 'n_ad1': 'n', 'k_ad1': 'k'})
    df = df[df['n'] >= 10].sort_values(by='mle', ascending=False)
    df = df[df['mle'] >= 0.10]
    print(df)
    print(df.shape[0])

    ctd1_handle = EmergeHandler(df_emerge = df)
    query = ctd1_handle.get(seq='GGCUUUCAAC')
    print(query)

    ctd1_bpe = EmergeBPE(
        df_emerge=df,
        kmax=6
    )
    ctd1_bpe.encode()

    ctd1_phenotype = ForestPhenotype(
        forest=ctd1_bpe.to_forest(),
        df_emerge=df
    )
    list_before = ctd1_phenotype.flatten()
    ctd1_phenotype.prune_by_enrichment()
    list_after = ctd1_phenotype.flatten()

    total_before = len(list_before)
    total_after = len(list_after)
    print(f'total before: {total_before}, total after: {total_after}')

    ctd1_phenotype.traverse(func=ctd1_phenotype.append_edits_bb)

    ctd1_phenotype.to_html(outfile='test.html', min_length=0)

if __name__ == "__main__":
    main()
