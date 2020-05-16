from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from cache import cache


@cache
def dummy_classifier(train_x, train_y, valid_x, valid_y, grid=False):
    """
    Creates and evaluates dummy classifier.

    :param train_x: train data
    :param train_y: train labels
    :param valid_x: validation test data
    :param valid_y: validation test labels
    :param grid: true for finding best params
    :return: dummy model fitted, predictions and accuracy
    """
    if not grid:
        dummy = DummyClassifier(strategy="uniform", random_state=42)
        dummy.fit(train_x, train_y)
        predictions = dummy.predict(valid_x)
        accuracy = accuracy_score(valid_y, predictions)
        return dummy, predictions, accuracy

    param_grid = {
        "strategy": ["stratified", "most_frequent", "uniform"],  # constant = most_frequent
    }
    grid_search = GridSearchCV(
        DummyClassifier(random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=-3,
        verbose=4,
        cv=5,
        refit=True
    )
    grid_search.fit(train_x, train_y)
    dummy = grid_search.best_estimator_
    predictions = dummy.predict(valid_x)
    accuracy = accuracy_score(valid_y, predictions)
    return dummy, predictions, accuracy


@cache
def bayes_classifier(train_x, train_y, valid_x, valid_y, grid=False):
    """
    Creates and evaluates GaussianNB classifier.

    :param train_x: train data
    :param train_y: train labels
    :param valid_x: validation test data
    :param valid_y: validation test labels
    :param grid: just for compatibility, ignored
    :return: GaussianNB model fitted, predictions and accuracy
    """
    gauss = GaussianNB()
    gauss.fit(train_x, train_x)
    predictions = gauss.predict(valid_x)
    accuracy = accuracy_score(valid_y, predictions)
    return gauss, predictions, accuracy


@cache
def tree_classifier(train_x, train_y, valid_x, valid_y, grid=False):
    """
    Creates and evaluates DecisionTree classifier.

    :param train_x: train data
    :param train_y: train labels
    :param valid_x: validation test data
    :param valid_y: validation test labels
    :param grid: just for compatibility, ignored
    :return: DecisionTree model fitted, predictions and accuracy
    """
    if not grid:
        dtree = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=100,
            min_samples_split=100
        )
        dtree.fit(train_x, train_y)
        predictions = dtree.predict(valid_x)
        accuracy = accuracy_score(valid_y, predictions)
        return dtree, predictions, accuracy

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_features": [None, "auto", "log2"],
        "max_depth": [20, 50, 100, 200, 300],
        "min_samples_split": [10, 100, 500, 1000],
        "min_samples_leaf": [10, 100, 200, 500, 1000]
    }
    grid_search = RandomizedSearchCV(
        DecisionTreeClassifier(),
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        n_jobs=-3,
        cv=5,
        refit=True,
        verbose=20,
        random_state=42,
    )

    grid_search.fit(train_x, train_y)
    dtree = grid_search.best_estimator_
    predictions = dtree.predict(valid_x)
    accuracy = accuracy_score(valid_y, predictions)
    return dtree, predictions, accuracy
