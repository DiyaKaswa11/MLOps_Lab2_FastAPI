"""Loading and splitting the Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    """Returns the Iris features (X) and labels (y)."""
    iris = load_iris()
    return iris.data, iris.target


def split_data(X, y, test_size=0.3, random_state=12):
    """Splits data into train and test sets, keeping class proportions equal (stratified)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
