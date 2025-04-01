# Aggregate Regression Patterns (ARP) Implementation

This repository contains an implementation of the **Aggregate Regression Patterns (ARP)** framework as described in the paper, as part of the Data Managment class:

**"Going Beyond Provenance: Explaining Query Answers with Pattern-based Counterbalances"**  
*Zhengjie Miao, Qitian Zeng, Boris Glavic, Sudeepa Roy*  
SIGMOD '19, June 30-July 5, 2019, Amsterdam, Netherlands.

The implementation is designed to analyze and explain outliers in aggregation queries using pattern-based counterbalances.

---

## Dataset

The dataset used in this project contains **crime data from Los Angeles** for the years **2020 to 2025**. The dataset includes the following attributes:

- **ID**: The Id of the crime.
- **DateAccour**: The Date of the crime.
- **AreaNumber**: The area where the crime occurred.
- **Type**: The type of crime (e.g., theft, assault, burglary).

The dataset is stored in a db file (`crimes.db`) and is used as input for the ARP framework.

## Prerequisites

Before running this project, make sure you have Python installed on your system. This project was developed with Python 3.x.

### Git LFS Setup (Required for Database File)

The database file is stored using Git Large File Storage (LFS). To properly clone it:

1. Install Git LFS:
   - Windows: Download and install from [git-lfs.github.com](https://git-lfs.github.com/)
   - macOS: Run
     ```bash
     brew install git-lfs
     ```
   - Linux: Run
     ```bash
     sudo apt-get install git-lfs
     ```

2. After installation, enable Git LFS:
```bash
git lfs install
```

3. After cloning the repository, pull the database file:
```bash
git lfs pull
```

### Required Libraries

Install the required Python libraries using pip:

```bash
pip install pandas
pip install scipy
pip install numpy
```

### Running the Program

Copy and paste the appropriate command in your terminal based on your operating system:

For Windows:
```
python main.py
```

For macOS/Linux:
```
python3 main.py
```

## Additional Information

The database file (`crimes.db`) will be automatically downloaded when you follow the Git LFS setup instructions above.

⚠️ **Important Performance Notes:**
- Clean your RAM or restart your PC before running the program for optimal performance
- The algorithm typically takes between 5-20 minutes to complete execution based on your system specifications
