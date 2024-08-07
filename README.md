# cellularlint-codes
### (The contents of the repository are currently under Artifact Evaluation from USENIX Security)

## Hardware Specifications
CellularLint was successfully run with the following hardware-
* CPU: AMD Ryzen Threadripper PRO 5965WX (24 core, 3.8 GHz)
* GPU: Nvidia RTX 3090 (24GB)
* Memory: 64GB.
* Additionally, we recommend 50 GB of available disk space. However, it may work with less.


## OS & Software Specifications
* CellularLint can be successfully run on Ubuntu 20.04 LTS and using python3. It should also run on Ubuntu 22.04 LTS and later stable versions.

## Installation
Please follow these steps to set up the environment-
(*For all the steps, we assume the current directory is:* ```cellularlint-codes```)
* Download the [pretrained models](https://zenodo.org/records/12199206) from Zenodo and place them under the ```Pretrained Models/``` directory.
* Download the [SNLI Train Dataset](https://zenodo.org/records/12249320) from Zenodo and place it under the ```Data/SNLI/``` directory. The validation and test datasets are already there.
* Run ```chmod 700 unpack.sh``` followed by ```./unpack.sh``` to unpack the pretrained models in the correct way.
* Run ```pip install -r requirements.txt``` to install the required packages. (Note: The requirements were generated using `pip freeze` and modified manually to consider only the required packages. If a package is missing or runs into a problem, you may remove the specific version number, and it should still work.)

## Running the experiments
1. The following experiment is to generate figures similar to **Figure 3** and **Figure 4** of the paper.

   From the main directory, run-
```python3 tokenizer_and_sim_matrix.py 4G```
and
```python3 tokenizer_and_sim_matrix.py 5G```
for 4G and 5G datasets, respectively. Each of these should generate one PDF and one PNG formatted image file (Thus, in total, 4 files are generated) in the main directory. The generated files are-
    - *4G_embedding_times.png*,
    - *heatmap_4G.pdf*,
    - *5G_embedding_times.png*, and
    - *heatmap_5G.pdf*.

    The PDF files can be compared to Figure 3 of the paper, and PNG files can be compared to Figure 4 of the paper.

2. The following experiment is to generate the models' performance metrics.
  
   Follow these instructions to train and use the language models- 
    * Run the notebooks **sequentially** in the following order. (If you are using Jupyter GUI, you may do `Kernel`>`Restart & Run All` for each of them)-
        - *train_bert.ipynb*,
        - *train_roberta.ipynb*,
        - *train_xlnet.ipynb*,
        - *phase_train_bert.ipynb*,
        - *phase_train_roberta.ipynb*, and
        - *phase_train_xlnet.ipynb*
      
    * From the ```eval/``` directory, run ```python3 -W ignore eval.py```. It should generate the metrics (See `output_metrics.txt` in the same directory) like Table 1 (one phase) of the paper.


## Citation
If you use the code or dataset used here, please cite our paper:
```
@article{rahman2024cellularlintsystematicapproachidentify,
      title={CellularLint: A Systematic Approach to Identify Inconsistent Behavior in Cellular Network Specifications}, 
      author={Mirza Masfiqur Rahman and Imtiaz Karim and Elisa Bertino},
      year={2024},
      eprint={2407.13742},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2407.13742}, 
}
```
