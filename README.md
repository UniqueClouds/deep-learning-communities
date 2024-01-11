# Deep Learning Communities

## Table of Contents
- [Deep Learning Communities](#deep-learning-communities)
  - [Table of Contents](#table-of-contents)
  - [Package Structure](#package-structure)
  - [Installation](#installation)
  - [Data](#data)
    - [Data Collection](#data-collection)
    - [Data Preprocessing](#data-preprocessing)
    - [Data Formats](#data-formats)
  - [Results](#results)
  - [Downloads](#downloads)
  - [Contact](#contact)
  - [License](#license)


## Package Structure
```sh
Deep_learning_communities
├── data/ #Data directory
│   ├── contributions # contribution_table of global/month/releases
│   ├── matching # 
│   ├── preprocessing/ # Cleaned and Processed data
│   ├── raw/ # Raw, unprocessed data
│   └── sampling # no need
├── outputs/ # Outpus from analysis
│   ├── figures # Generated graphics and figures
│   └── results # Results and other output files
├── src/ # Source code
│   ├── notebooks / # Jupyter notebooks for replication
│   ├── data_collection/ #Data collection scripts
│   ├── data_preprocessing/ # Data preprocessing scripts
│   ├── rq1_contributors_in_DLCommunities.py # Python Script for RQ1
│   ├── rq2_Evolution_of_DLCommunities.py # Python Script for RQ2
│   └── rq3_Impact_of_Community_Characteristics.py # Python Script for RQ3
├── README.md
└── requirements.txt
```
## Installation
- OS: Ubuntu 22.04(wsl2 on Windows) or Windows 11
- Software:
Python 3 (tested on Python )
- Required Python packages:
    ```sh
    pip install -r requirements.txt
    ```
- Node.js

## Data

Describe the sources, formats, and preprocessing methods of the data used in the project.
### Data Collection

We use `node.js` to crawl data through `GitHub API`, including `commits`, `issues`, `issue-events`, we maintain the original `json` format for data preprocessing.
The commits are from the main branch of the repository, and the issues and issue-events are from the main branch and the pull request of the repository.

The detailed data collection process is described in `DataCollection.md`.

The star number is sampled on the website [GitHub Stats
](https://vesoft-inc.github.io/github-statistics/)

Note: this may take a long time to crawl data, you are welcome to skip this step and use the data we provide directly.

### Data Preprocessing

**You can just run the `./src/notebooks/data_preprocessing.ipynb` to preprocess the data.**

We use `python` and `node.js`to preprocess the data, details are as follows:

- matching user login of commits
- bot exclusion: first use [`BoDeGHa`](https://github.com/mehdigolzadeh/BoDeGHa) to automatically detect bots, then manually check the results and remove false positives.
- generate final developer list
- contribution table generation

### Data Formats
The preprocessed data that following steps will use are stored in the `data/preprocessing` folder, the format is as follows:
- `commits`: `[hash,timestamp,name,email,LOC]`
- `issues`: `[issue_id,author,issue_type,timestamp,title,body]`
- `issue_events`: `[issue_id,actor,timestamp,type,body]`
- `releases_time`: `[releases, timestamp]`
- `stars`
  - `stars_monthly`:`[timestamp,stars]`
  - `stars_releases`: `[timestamp,stars]`
- `developers`: `[login,name,email,have login]`

The contribution table format for further analysis: `[login,commit_count,LOC,issue_count,central_degree]`

## Results

You can just follow the `./src/notebooks/results.ipynb` to get the results as it provides a detailed and executable procedure.

## Downloads

Because the crawled and processed data is too big, we provide the data from the [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10477123.svg)](https://doi.org/10.5281/zenodo.10477123), please download and unzip it in the `./data/` folder.

The raw data file structure is like this:
```sh
raw
├── commits
│   ├── pytorch
│   │   ├── pages # based on page number, store the github rest API, commits request information
│   │   │   ├── 1.json
│   │   │   ├── ...
│   │   ├── singleCommit  # based on hash, store all commit json source data
│   │   │   ├──9788a74da8fdba0675f0e67a0fbe55c3eb5dc486.json
│   │   │   ├──.... 
│   ├── pytorch.csv # store the processing results, the format is consistent with the original commits/pytorch.csv
│   ├── pytorch_manual.csv # store the mapping of [name, email] => login manually organized
│   ├── tensorflow
│   │   ├── pages  # based on page number, store the github rest API, commits request information
│   │   ├── singleCommit  # based on hash, store all commit json source data
│   ├── tensorflow.csv # store the processing results, the format is consistent with the original commits/pytorch.csv
│   └── tensorflow_manual.csv # store the mapping of [name, email] => login manually organized
├── issue-events
│   ├── pytorch # store all issue/pr event
│   │   ├── 1.json
│   │   ├── 2.json
│   │   ├── ...
│   ├── pytorch.csv # store the processing results, the format is consistent with the original issue-events/pytorch.csv
│   ├── pytorch_developer.txt # store all developers corresponding to issue-event
│   ├── tensorflow # store all issue/pr event
│   │   ├── 1.json
│   │   ├── 2.json
│   │   ├── ...
│   ├── tensorflow.csv # store the processing results, the format is consistent with the original issue-events/pytorch.csv
│   ├── tensorflow_developer.txt # store all developers corresponding to issue-event
├── issues
│   ├── pytorch
│   │   ├── pages  # based on page number, store the github rest API, issues request information
│   ├── pytorch.csv # store the processing results, the format is consistent with the original issues/pytorch.csv
│   ├── tensorflow
│   │   ├── pages  # based on page number, store the github rest API, issues request information
│   ├── tensorflow.csv # store the processing results, the format is consistent with the original issues/pytorch.csv
├── matching
│   ├── pytorch.csv # store the matching results of pytorch
│   └── tensorflow.csv  # store the matching results of tensorflow
├── pytorch_developers.csv # final developer file
└── tensorflow_developers.csv # final developer file

```

## Contact
If you have any questions regarding this paper, please do not hesitate to contact us:

## License
Apache License 2.0
