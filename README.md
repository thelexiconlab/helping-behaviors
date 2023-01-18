# assistance-games

Repository for storing data and analysis scripts for the helping behaviors project

## data structure

1. `e1 data` contains the raw gameplay data from 100 dyads (N = 200)
2. `e1 demo` contains the demographic information 

## code structure

1. There are 4 key .py files:
- `utils.py` contains base functions to create goalspace, compute utilities, etc. 
- `agents.py` contains the functions for the helper and demonstrator/architect models
- `process_data.py` contains functions to obtain move sequences, ranks, etc. from the raw data
- `optimizer.py` contains functions to obtain the best-fitting parameter values for different agent models

2. There is also a notebook `assistance_demo.ipynb` that contains a demonstration of how to look at the data and obtain outputs from different agents/functions.

3. Finally, there is a `.Rmd` file that contains the analyses based on results obtained from the python files.




