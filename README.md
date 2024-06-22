# cellularlint-codes
### (The contents of the repository is currently under Artifact Evaluation from USENIX Security)
## Installation
Please follow these steps to set up the environment-
(*For all the steps, we assume the current directory is:* ```cellularlint-codes```)
* Download the [pretrained models](https://zenodo.org/records/12199206) from Zenodo and place them under the ```Pretrained Models/``` directory.
* Download the [SNLI Train Dataset](https://zenodo.org/records/12249320) from Zenodo and place it under the ```Data/SNLI/``` directory. The validation and test dataset are already there.
* Run ```chmod 700 unpack.sh``` followed by ```./unpack.sh``` to unpack the pretrained models in the correct way.
* Run ```pip install -r requirements.txt``` to install the required packages.

## Running the experiments
1. From the main directory, run-
```python3 tokenizer_and_sim_matrix.py 4G```
and
```python3 tokenizer_and_sim_matrix.py 5G```
for 4G and 5G datasets, respectively. Each of these should generate one PDF and one PNG formatted image file (Thus, in total 4 files are generated) in the main directory. The generated files are-
    - *4G_embedding_times.png*,
    - *heatmap_4G.pdf*,
    - *5G_embedding_times.png*, and
    - *heatmap_5G.pdf*.
  
The PDF files can be compared to Figure 3 of the paper, and PNG files can be compared to Figure 4 of the paper.

2. Follow these instructions to train and use the language models- 
    * Run the notebooks sequentially in the following order-
        - *train_bert.ipynb*,
        - *train_roberta.ipynb*,
        - *train_xlnet.ipynb*,
        - *phase_train_bert.ipynb*,
        - *phase_train_roberta.ipynb*, and
        - *phase_train_xlnet.ipynb*
      
    * From the ```eval/``` directory, run ```python3 -W ignore eval.py```. It should generate the metrics like Table 1 (one phase) of the paper.
