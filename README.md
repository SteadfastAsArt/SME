# Simple Motif Embedding

# Introduction

# Using
+ detect_contract.py
    + input: original graph
    + output: intermediate results
+ training_method.py
    + input: graph
    + output: `graph_name(0).method.emb` embeddings(0 as original and 1 as intermediate)
+ release_motif.py
    + output: `graph_name1.method.emb` (1 as final)
+ performance analysis.py<br>
to do the task of network reconstruction
+ SME.py
    + a wrapper file containing all the steps above

# parameter setting
+ LE
+ Deepwalk<br>

| dim | walk_length| walks_node| window| workers|
|:---:|:---:|:---:|:---:|:---:|
| 128 | 80 | 10 | 5 | 12 |

+ node2vec<br>

| dim | walk_length| walks_node| window |workers|p|q|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 80 | 10 | 5 | 12 | 2 | 0.5 |

+ LINE
+ GAEs
