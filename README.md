# MATH5472-Project
MATH5472 Project: a Python implementation of the paper ["Weighted Low Rank Matrix Approximation and Acceleration"](https://arxiv.org/abs/2109.11057)

## Codes
This repo contains the following codes:
- `wlrma_python.py`: contains the implementation of the SVD version of the algorithms solving WLRMA problems, corresponding to Algorithms 1-3 in the report.
- `wlrma_als_python.py`: contains the implementation of the non-SVD version of the algorithms solving WLRMA problems on high-dimensional data, corresponding to Algorithms 4-6 in the report.
- `simulation.ipynb`: Jupyter Notebook that runs the simulation study.
- `movielens_example.ipynb`: Jupyter Notebook that runs the MovieLens example.

## Other Files
- `data`: the folder contains the data needed to run MovieLens example.
- `MATH5472 Project Report.pdf`: the report for this project. 

## How to Run the Demos
Run `simulation.ipynb` for the simulation study and `movielens_example.ipynb` for the MovieLens example. The latter may take a long time to execute. 
