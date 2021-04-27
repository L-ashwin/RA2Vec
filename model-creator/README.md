# Model Creator
The model-creator.ipynb notebook contains functions to create embeddings for n-grams of amino acid sequences.  
There is a option to create both simple [ProtVec](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287) 
  (using all 20 alphabets) and [RA2Vec](https://dl.acm.org/doi/10.1145/3388440.3414925) (Reduced Alphabet) embeddings.  
This notebook uses protein sequences from [UniProt](https://www.uniprot.org/)'s SWISS-PROT database (a curated protein sequence database) as a corpus to generate the embeddings.
  you can download the compressed file from this [link](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz), 
  **extract and add uniprot_sprot.fasta in data directory before running the notebook**.
