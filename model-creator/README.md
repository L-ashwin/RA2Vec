# Model Creator
The model-creator.ipynb notebook contains functions to create embeddings for n-grams of amino acid sequences.  
There is a option to create both simple [ProtVec](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287) 
  (using all 20 alphabets) and [RA2Vec](https://dl.acm.org/doi/10.1145/3388440.3414925) (Reduced Alphabet) embeddings.  
This notebook uses protein sequences from [UniProt](https://www.uniprot.org/)'s SWISS-PROT database (a curated protein sequence database) as a corpus to generate the embeddings.
  you can download the compressed file from this [link](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz), 
  **extract and add uniprot_sprot.fasta in data directory before running the notebook**.   
  
  ## Steps involved in Creating RA2Vec Models
  1. Define model parameters  
    - *kGrams* : word length to be used  
    - *vecSize* : size of embedding vector to be generated  
    - *window* : window parameter from the skip-gram algorithm (number of words on either side to be considered)   
    - *trans* : translation dictionary to be used (which amino acid grouping to be used ex. Hypropathy, Conformational Similarity)  
  2. Prepare the Corpus (*SentenceGenerator*)  
    - get the sequences from UniProt  
    - remove sequences that contain uncommon amino acids  
    - translate sequences from initial 20 letter representation to reduced alphabet representation  
    - create sentences from each of the sequences with given kGram size.  
    - *SentenceGenerator* object yields one sentence at a time  
  3. Word2Vec model training  
    - skip-gram algorithm is used to generate the distributed representation.  
