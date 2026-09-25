"""
trains and compares the regression models for predicting final_test.

the split happens before anything is fitted, and every step that learns from
the data (imputing attendance_rate, one hot encoding, scaling) sits inside a
Pipeline. that way cross validation refits them on each fold and nothing from
the test set leaks into training.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data_preparation import PROJECT_ROOT, load_config, prepare_data


SCORING = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def split_features_target(df, config):
    y = df[config["target"]]
    X = df.drop(columns=[config["target"]] + config["drop_for_modelling"])
    return X, y


def build_preprocessor(X, preprocessing):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric = Pipeline([
        ("impute", SimpleImputer(
            strategy=preprocessing["numeric_impute_strategy"],
            add_indicator=preprocessing["add_missing_indicator"],
        )),
        # only the linear and distance based models need this, but scaling the
        # trees too costs nothing and keeps one preprocessor for every model
        ("scale", StandardScaler()),
    ])

    # sparse_output=False because HistGradientBoosting cannot take sparse input.
    # handle_unknown="ignore" so an unseen category at predict time becomes all
    # zeros instead of an error
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def get_models(random_state):
    return {
        # not a real model, just the score to beat. predicts the mean every time
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        # one tree on its own, to show what the ensembles are built out of
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        # many trees on random subsets, averaged
        "Random Forest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        # trees built one after another, each fixing the last one's errors
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),
    }


def cross_validate_models(pipelines, X_train, y_train, evaluation):
    """scores whatever pipelines it is handed, tuned or not.

    this runs after tuning so that the cross validation numbers and the test
    numbers describe the same estimator. scoring the defaults here and the
    tuned version on the test set would make the two columns incomparable.
    """
    cv = KFold(
        n_splits=evaluation["cv_folds"],
        shuffle=True,
        random_state=evaluation["random_state"],
    )

    rows = []
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)

        rows.append({
            "model": name,
            # sklearn returns these negated so that higher is always better
            "cv_mae": -scores["test_mae"].mean(),
            "cv_mse": -scores["test_mse"].mean(),
            "cv_rmse": -scores["test_rmse"].mean(),
            "cv_rmse_std": scores["test_rmse"].std(),
            "cv_r2": scores["test_r2"].mean(),
            "fit_seconds": scores["fit_time"].mean(),
        })
        print(f"  {name:<20} rmse {rows[-1]['cv_rmse']:6.3f}  r2 {rows[-1]['cv_r2']:6.3f}")

    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def tune_models(models, preprocessor, X_train, y_train, tuning, evaluation):
    """grid search the models that have a grid in config, leave the rest alone."""
    cv = KFold(
        n_splits=evaluation["cv_folds"],
        shuffle=True,
        random_state=evaluation["random_state"],
    )

    tuned, best_params, all_scores = {}, {}, []
    for name, model in models.items():
        grid = tuning["grids"].get(name)
        pipe = Pipeline([("prep", clone(preprocessor)), ("model", model)])

        if not grid:
            tuned[name] = pipe
            continue

        search = GridSearchCV(
            pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        search.fit(X_train, y_train)

        tuned[name] = search.best_estimator_
        best_params[name] = search.best_params_
        print(f"  {name:<20} {search.best_params_}")

        # keep every combination that was tried, not just the winner
        scores = pd.DataFrame(search.cv_results_)
        scores["model"] = name
        scores["cv_rmse"] = -scores["mean_test_score"]
        scores["cv_rmse_std"] = scores["std_test_score"]
        scores["params"] = scores["params"].astype(str)
        all_scores.append(
            scores[["model", "params", "cv_rmse", "cv_rmse_std", "rank_test_score"]]
        )

    grid_scores = (
        pd.concat(all_scores).sort_values(["model", "cv_rmse"]).reset_index(drop=True)
        if all_scores
        else pd.DataFrame()
    )

    return tuned, best_params, grid_scores


def evaluate_on_test(fitted, X_test, y_test):
    """the test set is touched once, here, after every choice has been made."""
    rows = []
    for name, pipe in fitted.items():
        pred = pipe.predict(X_test)
        rows.append({
            "model": name,
            "test_mae": mean_absolute_error(y_test, pred),
            "test_mse": mean_squared_error(y_test, pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "test_r2": r2_score(y_test, pred),
        })

    return pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)


def error_profile(pipe, X_test, y_test):
    """how far off the chosen model is, in marks.

    the four headline metrics are all summaries of the same errors. this shows
    the errors themselves, which is easier to explain to someone who does not
    already know what an rmse is.
    """
    errors = np.abs(pipe.predict(X_test) - y_test.values)

    rows = [
        ("students in the test set", len(errors)),
        ("half the students are missed by less than", np.median(errors)),
        ("average miss (this is the MAE)", errors.mean()),
        ("root mean squared error (the RMSE)", np.sqrt((errors ** 2).mean())),
        ("9 in 10 students are missed by less than", np.percentile(errors, 90)),
        ("99 in 100 students are missed by less than", np.percentile(errors, 99)),
        ("worst single miss", errors.max()),
    ]
    rows += [
        (f"share predicted within {t} marks", (errors <= t).mean() * 100)
        for t in (2, 5, 10, 15)
    ]

    return pd.DataFrame(rows, columns=["measure", "value"])


def feature_importances(pipe, name):
    """coefficients for the linear models, impurity importance for the trees."""
    model = pipe.named_steps["model"]
    features = pipe.named_steps["prep"].get_feature_names_out()

    if hasattr(model, "coef_"):
        values = model.coef_
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return None

    return (
        pd.DataFrame({"feature": features, name: values})
        .assign(sort_key=lambda d: d[name].abs())
        .sort_values("sort_key", ascending=False)
        .drop(columns="sort_key")
        .reset_index(drop=True)
    )


def main():
    config = load_config()

    df = prepare_data(config)
    X, y = split_features_target(df, config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )
    print(f"train {len(X_train)} rows, test {len(X_test)} rows")
    # the baseline model predicts this same number for every student
    print(f"training average final_test: {y_train.mean():.2f}\n")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(
        X, config["preprocessing"]
    )
    print(f"numeric: {numeric_cols}")
    print(f"categorical: {categorical_cols}\n")

    models = get_models(config["evaluation"]["random_state"])

    best_params = {}
    if config["tuning"]["enabled"]:
        print("tuning")
        fitted, best_params, grid_scores = tune_models(
            models, preprocessor, X_train, y_train,
            config["tuning"], config["evaluation"],
        )
    else:
        grid_scores = pd.DataFrame()
        fitted = {
            name: Pipeline([("prep", clone(preprocessor)), ("model", model)])
            for name, model in models.items()
        }

    print(f"\ncross validation on the training set ({config['evaluation']['cv_folds']} folds)")
    cv_results = cross_validate_models(fitted, X_train, y_train, config["evaluation"])

    for pipe in fitted.values():
        pipe.fit(X_train, y_train)

    test_results = evaluate_on_test(fitted, X_test, y_test)

    comparison = cv_results.merge(test_results, on="model").sort_values("test_rmse")
    print("\nmodel comparison")
    print(comparison.round(3).to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    print(f"\nbest by test rmse: {best_name}")

    profile = error_profile(fitted[best_name], X_test, y_test)
    print(f"\nhow far off {best_name} is, in marks")
    print(profile.round(2).to_string(index=False))

    importances = feature_importances(fitted[best_name], best_name)
    if importances is not None:
        print(f"\ntop features for {best_name}")
        print(importances.head(15).round(4).to_string(index=False))

    results_dir = PROJECT_ROOT / config["evaluation"]["results_dir"]
    results_dir.mkdir(exist_ok=True)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    if importances is not None:
        importances.to_csv(results_dir / "feature_importances.csv", index=False)
    if best_params:
        pd.DataFrame(
            [{"model": k, "best_params": str(v)} for k, v in best_params.items()]
        ).to_csv(results_dir / "best_params.csv", index=False)
    if not grid_scores.empty:
        grid_scores.to_csv(results_dir / "grid_search_results.csv", index=False)
    profile.to_csv(results_dir / "error_profile.csv", index=False)
    print(f"\nsaved to {results_dir}")


if __name__ == "__main__":
    main()
