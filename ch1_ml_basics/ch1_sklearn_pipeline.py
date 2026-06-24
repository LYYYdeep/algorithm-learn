import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def main():
    data = {
        "name": [
            "Alice", "Bob", "Cindy", "David", "Eva", "Frank", "Grace", "Helen",
            "Ian", "Jack", "Kate", "Leo", "Mia", "Nick", "Olivia", "Peter",
        ],
        "city": [
            "Beijing", "Shanghai", "Beijing", "Shenzhen",
            "Shanghai", "Beijing", "Shenzhen", "Beijing",
            "Shanghai", "Shenzhen", "Beijing", "Shanghai",
            "Shenzhen", "Beijing", "Shanghai", "Shenzhen",
        ],
        "education": [
            "HighSchool", "Bachelor", "Bachelor", "Master",
            "Master", "PhD", "HighSchool", "Bachelor",
            "Master", "PhD", "HighSchool", "Bachelor",
            "Master", "Bachelor", "PhD", "Master",
        ],
        "study_hours": [
            0.5, 1.0, 1.5, 2.0,
            2.5, 3.0, 3.5, None,
            4.5, 5.0, 1.0, 1.5,
            2.0, 3.0, 4.0, 5.0,
        ],
        "exercises": [
            5, 8, 10, 15,
            18, 25, 30, 35,
            40, 45, 3, 6,
            None, 12, 20, 30,
        ],
        "passed": [
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 0, 0,
            0, 1, 1, 1,
        ],
    }

    df = pd.DataFrame(data)

    feature_columns = ["city", "education", "study_hours", "exercises"]
    target_column = "passed"

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    numeric_features = ["study_hours", "exercises"]
    nominal_features = ["city"]
    ordinal_features = ["education"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[["HighSchool", "Bachelor", "Master", "PhD"]]
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("nominal", nominal_transformer, nominal_features),
        ("ordinal", ordinal_transformer, ordinal_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    print("Evaluation")
    print("----------")
    print("X_test:")
    print(X_test)
    print("y_test:", y_test.to_numpy())
    print("y_prob:", y_prob)
    print("y_pred:", y_pred)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    new_students = pd.DataFrame({
        "city": ["Beijing", "Shanghai", "Hangzhou"],
        "education": ["Bachelor", "Master", "PhD"],
        "study_hours": [1.0, 3.0, 5.0],
        "exercises": [5, 20, 40],
    })

    new_prob = model.predict_proba(new_students)[:, 1]
    new_pred = model.predict(new_students)

    print()
    print("New Students:")
    print(new_students)

    print()
    print("New Predictions:")
    print("Pass probabilities:", new_prob)
    print("Predictions:", new_pred)


if __name__ == "__main__":
    main()
