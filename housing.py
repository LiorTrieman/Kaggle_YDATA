
import pandas as pd

class FeatureExtracter:
    """Extracts features used for ML regression."""

    # Columns to cast as categories.
    cols_cast_category = [
        "MSSubClass"
    ]

    # Columns to fillna with "None" string.
    cols_fillna_None_str = [
        "PoolQC", 
        "MiscFeature", 
        "Alley", 
        "Fence", 
        "FireplaceQu", 
        "GarageType", 
        "GarageFinish", 
        "GarageQual", 
        "GarageCond", 
        "BsmtExposure", 
        "BsmtFinType2", 
        "BsmtFinType1", 
        "BsmtCond", 
        "BsmtQual", 
        "MasVnrType",
        "Electrical",

        # NaNs found only in test set.
        "MSZoning",
        "Exterior1st",
        "Exterior2nd",
        "KitchenQual",
        "Functional",
        "SaleType",
    ]

    # Columns to fillna with 0.
    cols_fillna_0 = [
        "LotFrontage", 
        "MasVnrArea",

        # NaNs found only in test set.
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "GarageCars",
    ]

    # Columns to drop.
    cols_drop = [
        # Features highly correlated to other features.
        "1stFlrSF",
        "GarageArea",
        "GarageYrBlt",
        "TotRmsAbvGrd",

        # Categorical features with >99% of data in only 1 category.
        "Street",
        "Utilities",
    ]

    # Features to binarize.
    #   Key is columns name, value is value to consider True (all others are False).
    binarize_features_by_true_value = {
        "Condition2": "Norm",
        "RoofMatl": "CompShg",
        "Heating": "GasA",
    }

    #   Key is columns name, value is value to consider False (all others are True).
    binarize_features_by_false_value = {
        "LowQualFinSF": 0,
        "3SsnPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
    }

    def __init__(self):
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._all_categories: list[list[str]] = []

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = self.transform(X)

        self._num_cols = X.select_dtypes(include="number").columns.to_list()
        self._cat_cols = X.select_dtypes(exclude="number").columns.to_list()
        self._all_categories = [sorted(list(X[col].unique())) for col in self._cat_cols]

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._cast_types(X)
        X = self._fillna(X)
        X = self._drop_columns(X)
        X = self._transform_date_sold(X)
        X = self._binarize_features(X)
        self._validate_no_nans(X)
        return X

    def get_numeric_feature_names(self) -> list[str]:
        return self._num_cols

    def get_categorical_feature_names(self) -> list[str]:
        return self._cat_cols

    def get_category_lists(self) -> list[list[str]]:
        """Return contains sorted listed of all categories found in training set for each categorical feature."""
        return self._all_categories

    def _cast_types(self, X):
        for col in self.cols_cast_category:
            X[col] = X[col].astype("category")
        return X

    def _fillna(self, X):
        X[self.cols_fillna_None_str] = X[self.cols_fillna_None_str].fillna("None")
        X[self.cols_fillna_0] = X[self.cols_fillna_0].fillna(0)
        return X

    def _drop_columns(self, X):
        return X.drop(columns=self.cols_drop)

    def _binarize_features(self, X):
        for col, true_val in self.binarize_features_by_true_value.items():
            X[f"{col}_is_{true_val}"] = X[col] == true_val

        for col, false_val in self.binarize_features_by_false_value.items():
            X[f"{col}_not_{false_val}"] = X[col] != false_val

        # Drop original columns.
        return X.drop(
            columns=list(self.binarize_features_by_true_value.keys())
                + list(self.binarize_features_by_false_value.keys()))

    def _transform_date_sold(self, X):
        """Replace MoSold and YrSold with MonthNumSold (month after Jan 2000)."""
        def get_months_after_start_year(dt, start_year=2000):
            return (dt.year - start_year) * 12 + dt.month
        
        date_sold = pd.to_datetime(X['MoSold'].astype(str) + '/' + X['YrSold'].astype(str), format="%m/%Y")
        X["MonthNumSold"] = date_sold.apply(get_months_after_start_year)

        # Drop original columns.
        return X.drop(columns=['MoSold', 'YrSold'])

    def _validate_no_nans(self, X):
        if X.isna().any().any():
            raise ValueError(f"NaNs present for the following IDs: {list(X.index[X.isna().any(axis=1)])} "
                f"and the following columns: {list(X.columns[X.isna().any(axis=0)])}")
