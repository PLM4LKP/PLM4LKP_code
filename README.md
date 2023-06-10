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
 | |- relatedness.ipynb     - experiment file
 |- utils
 | |- duplicate-checker.py  - to check whether there is duplicate data in dataset
 |- README.md
```

# Run
- clone this repository
- download the dataset from https://github.com/maxxbw54/ESEM2018 and put the csv files to PLM4LKP/data/raw
- open relatedness.ipynb with your IDE
- change some const configure variable and run cells you need