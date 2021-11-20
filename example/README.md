## EXPLAIN

The description of each file is as follows:

- 2cbla.fasta
  - Target sequence file that needs domain decomposition.
- 2cbla.npz
  - The predicted distance file of the target sequence. In DomBpred, the file comes from trRosetta, and the dimension of the trRosetta predicted distance file is L×L×37, while the dimension required by DomBpred is L×L×1. The conversion of the dimension can be through the program provided by trRosetta, use the sample refer to https://github.com/iobio-zjut/DomBpred#readme.
- 2cbla.ss2
  - The predicted secondary structure file of the target sequence. In DomBpred, the file comes from PSIPRED and does not need to be converted.
- 2cbla.txt
  - The MSA file of the target sequence, in DomBpred, the file is obtained by searching the SDSL database using jackhmmer, and the search example refers to https://github.com/iobio-zjut/DomBpred#readme.
- 2cbla_trRosetta.npz
  - The predicted distance file of the target sequence. This file is the result output by trRosetta without conversion. Its dimension is L×L×37, where L is the length of the target sequence. Due to GitHub upload restrictions, please download and view through http://zhanglab-bioinf.com/DomBpred/materials/.

