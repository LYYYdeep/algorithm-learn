import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    print("Original DataFrame:")
    print(df)

    print()
    print("Missing values:")
    print(df.isnull().sum())

    df["study_hours_missing"] = df["study_hours"].isnull().astype(int)
    df["exercises_missing"] = df["exercises"].isnull().astype(int)

    df["study_hours"] = df["study_hours"].fillna(df["study_hours"].mean())
    df["exercises"] = df["exercises"].fillna(df["exercises"].median())

    education_mapping = {
        "HighSchool": 0,
        "Bachelor": 1,
        "Master": 2,
        "PhD": 3,
    }

    df["education_encoded"] = df["education"].map(education_mapping)

    df = pd.get_dummies(
        df,
        columns=["city"],
        prefix="city",
        dtype=int,
    )

    feature_columns = [
        "study_hours",
        "exercises",
        "study_hours_missing",
        "exercises_missing",
        "education_encoded",
        "city_Beijing",
        "city_Shanghai",
        "city_Shenzhen",
    ]

    X = df[feature_columns]
    y = df["passed"]

    print()
    print("Processed DataFrame:")
    print(df)

    print()
    print("Features X:")
    print(X)

    print()
    print("Label y:")
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()

    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    print()
    print("Evaluation")
    print("----------")
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
        "name": ["Queen", "Robert", "Susan"],
        "city": ["Beijing", "Shanghai", "Shenzhen"],
        "education": ["Bachelor", "Master", "PhD"],
        "study_hours": [1.0, 3.0, 5.0],
        "exercises": [5, 20, 40],
    })

    new_students["study_hours_missing"] = new_students["study_hours"].isnull().astype(int)
    new_students["exercises_missing"] = new_students["exercises"].isnull().astype(int)

    new_students["study_hours"] = new_students["study_hours"].fillna(df["study_hours"].mean())
    new_students["exercises"] = new_students["exercises"].fillna(df["exercises"].median())

    new_students["education_encoded"] = new_students["education"].map(education_mapping)

    new_students = pd.get_dummies(
        new_students,
        columns=["city"],
        prefix="city",
        dtype=int,
    )

    for col in ["city_Beijing", "city_Shanghai", "city_Shenzhen"]:
        if col not in new_students.columns:
            new_students[col] = 0

    new_X = new_students[feature_columns]
    new_X_scaled = scaler.transform(new_X)

    new_prob = model.predict_proba(new_X_scaled)[:, 1]
    new_pred = model.predict(new_X_scaled)

    print()
    print("New Students:")
    print(new_students)

    print()
    print("New Predictions:")
    print("Pass probabilities:", new_prob)
    print("Predictions:", new_pred)


if __name__ == "__main__":
    main()
