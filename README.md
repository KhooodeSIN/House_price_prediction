# House Price Prediction

Predict the median house price (`MEDV`) from Boston housing features using a tuned **Gradient Boosting** regression model.

## Dataset

The project uses `train.csv`. Each row is one housing area.

| Column | Description |
|---|---|
| `ID` | Unique row label (not used by the model) |
| `CRIM` | Per capita crime rate by town |
| `ZN` | Share of residential land zoned for large lots |
| `INDUS` | Share of non-retail business acres |
| `CHAS` | Charles River dummy (1 if tract borders the river) |
| `NOX` | Nitric oxides concentration |
| `RM` | Average number of rooms |
| `AGE` | Share of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to Boston employment centers |
| `RAD` | Accessibility index to radial highways |
| `TAX` | Property-tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | `1000(Bk - 0.63)^2` town composition measure |
| `LSTAT` | Percent lower status of the population |
| `MEDV` | Median value of owner-occupied homes in $1000s (**target**) |

## What the notebook does

1. Loads `train.csv`
2. Checks for missing values
3. Removes `ID` and cleans empty rows
4. Builds useful extra features (`RM2`, `LSTAT2`, `RM_LSTAT`, `log_CRIM`, `TAX_PTRATIO`)
5. Splits the data into train and validation sets
6. Trains a tuned `GradientBoostingRegressor`
7. Reports **R²**, **MSE**, and average error
8. Plots actual price vs predicted price

## Results

On the held-out validation houses (20% of the data):

| Metric | Score |
|---|---|
| R² | about **89.3%** |
| MSE | about **11.4** |
| Average error | about **$2,140** |

## How to run

1. Install Python 3.10+
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open `house_analysis.ipynb` in Jupyter or VS Code / Cursor
4. Run all cells from top to bottom

## Project files

```text
House_price_prediction/
├── house_analysis.ipynb   # main analysis and model notebook
├── train.csv              # housing data
├── requirements.txt       # Python packages
└── README.md              # this file
```

## Model notes

- This is a **supervised regression** problem because the target `MEDV` is a continuous price.
- Gradient boosting was chosen because it handled non-linear price patterns better than a simple linear model on this dataset.
- Extra features and hyperparameter tuning were used to raise R² and lower MSE.
