# Aggregate Regression Patterns (ARP) Implementation

This repository contains an implementation of the **Aggregate Regression Patterns (ARP)** framework as described in the paper:

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
