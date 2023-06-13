# PLM4LKP
Implementation of the paper "Leveraging Pre-trained Language Models to Linkable Knowledge Prediction in Stack Overflow"

# Directory
```
- PLM4LKP
 |- data
 | |- raw   - csv data
 | | |- medium_link_prediction_noClue_shuffled_test.csv
 | | |- medium_link_prediction_noClue_shuffled_train.csv
 |- src
 | |- models                - some models are from our another experiment and reused
 | |- pre-training          - for TAPT
 | |- relatedness.ipynb     - experiment file with some models
 |- utils
 | |- duplicate-checker.py  - to check whether there is duplicate data in dataset
 |- README.md
```

# Dependency
```
python == 3.9.0
cuda == 12.0
jupyter_core == 4.9.1
IPython == 7.21.0

numpy == 1.20.1
pandas == 1.3.3
tqdm == 4.65.0
transformers == 4.28.1
torch == 1.13.1+cu117
```

# Dataset
- We utilized the dataset available at https://github.com/maxxbw54/ESEM2018. The dataset can be downloaded from its README.md file via a Google Drive link.
- Please place the CSV files into the folder PLM4LKP/data/raw.

# Run
- Open the relatedness.ipynb file using your Integrated Development Environment (IDE).
- Alter some constant configuration variables and execute the necessary cells.

# Seeking for help
If you encounter any issues while using our code, please submit an issue. We will respond as soon as possible. Issues written in either Chinese or English are welcome.