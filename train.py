import argparse
import joblib

from evaluate import load_train_dev_test, rmse

from transforms import FeatureProjection, TargetEncoder
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/home-listings-subset.csv')
parser.add_argument('--random_state', type=int, default=42)
parser.add_argument('--n_estimators', type=int, default=25)
parser.add_argument('--max_depth', type=int, default=None)
parser.add_argument('--save_model', action='store_true')

args = parser.parse_args()

(X_train, y_train), (X_dev, y_dev) = load_train_dev_test(args.data_path, args.random_state)

NUMERICAL_COLUMNS = [
'CDOM',
'LotSizeAreaSQFT',
'SqFtTotal',
]

features_pipe =  make_union(
    make_pipeline(
        FeatureProjection(NUMERICAL_COLUMNS),
        SimpleImputer() # There was no null data on the original dataset, but can be useful for new data
    ),
    TargetEncoder('BathsTotal'),
    TargetEncoder('BedsTotal'),
    TargetEncoder('ElementarySchoolName'),
    TargetEncoder('StructuralStyle', min_freq=1)

)

rf_pipe = make_pipeline(
    features_pipe,
    RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state)
)

rf_pipe.fit(X_train, y_train)

print('Train RMSE:', rmse(y_train, rf_pipe.predict(X_train)))
print('Dev RMSE:', rmse(y_dev, rf_pipe.predict(X_dev)))

if args.save_model:
    joblib.dump(rf_pipe, 'models/model.joblib')
