# cellularlint-codes
## Installation
Please follow these steps to set up the environment-
(*Assuming the current directory is:* ```cellularlint-codes```)
* Download the [pretrained models](https://zenodo.org/records/12199206) from Zenodo and place them under the ```Pretrained Models/``` directory.
* Download the [SNLI Train Dataset](https://zenodo.org/records/12249320) from Zenodo and place it under the ```Data/SNLI/``` directory. The validation and test dataset are already there.
* Run ```chmod 700 unpack.sh``` followed by ```./unpack.sh``` to place the pretrained models in the correct directory.
* Run ```pip install -r requirements.txt``` to install the required packages.
