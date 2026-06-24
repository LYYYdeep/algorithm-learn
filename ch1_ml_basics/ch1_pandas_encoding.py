import pandas as pd


def main():
    data = {
        "name": ["Alice", "Bob", "Cindy", "David", "Eva", "Frank"],
        "city": ["Beijing", "Shanghai", "Beijing", "Shenzhen", "Shanghai", "Beijing"],
        "education": ["Bachelor", "Master", "HighSchool", "PhD", "Bachelor", "Master"],
        "study_hours": [1.0, 3.0, 2.0, 5.0, 4.0, 3.5],
        "passed": [0, 1, 0, 1, 1, 1],
    }

    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    education_mapping = {
        "HighSchool": 0,
        "Bachelor": 1,
        "Master": 2,
        "PhD": 3,
    }

    df["education_encoded"] = df["education"].map(education_mapping)

    print()
    print("After ordinal encoding:")
    print(df)

    encoded_df = pd.get_dummies(
        df,
        columns=["city"],
        prefix="city",
        dtype=int,
    )

    print()
    print("After one-hot encoding:")
    print(encoded_df)

    feature_columns = [
        "study_hours",
        "education_encoded",
        "city_Beijing",
        "city_Shanghai",
        "city_Shenzhen",
    ]

    X = encoded_df[feature_columns]
    y = encoded_df["passed"]

    print()
    print("X:")
    print(X)

    print()
    print("y:")
    print(y)


if __name__ == "__main__":
    main()
