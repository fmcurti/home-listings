# Home Listings price prediction

### Creating environment
```
python -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the project

To run the training script

```
train.py [-h] [--data_path DATA_PATH] [--random_state RANDOM_STATE] [--n_estimators N_ESTIMATORS] [--max_depth MAX_DEPTH] [--save_model]
```

Notebooks used for exploring and cleaning the data can be run in order

### Future work
* A better split could be created by using date as criteria insted of a random one.
* The creation of a test set alongside the dev one could make for a better validation technique, allowing to do hyperparameter search and avoiding overfitting.
