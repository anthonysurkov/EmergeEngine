# emerge_vis.py
# eventually should abstract functions to make EmergeForest.to_html() less of a
# clusterfuck
# for now, want to reimplement heat map

bases = ['A','C','G','U']
seq = 'GCGGCGAGC'
keep = set(range(5,8))

snps = []
for i, wt in enumerate(seq):
    if i in keep:
        continue
    for b in bases:
        if b == wt:
            continue
        snps.append(seq[:i] + b + seqs[i+1:]
