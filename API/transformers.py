import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class MapColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, mapping=None):
        self.column = column
        self.mapping = mapping or {
            # Organic
            "organic": "organic",
            "(none)": "organic",
            "referral": "organic",

            # CPC
            "cpc": "paid_cpc",
            "yandex_cpc": "paid_cpc",
            "google_cpc": "paid_cpc",

            # CPM
            "cpm": "paid_cpm",
            "CPM": "paid_cpm",

            # Email
            "email": "email",
            "outlook": "email",

            # Social / SMM
            "push": "social",
            "smm": "social",
            "stories": "social",
            "post": "social",
            "blogger_channel": "social",
            "blogger_stories": "social",
            "blogger_header": "social",
            "tg": "social",
            "fb_smm": "social",
            "vk_smm": "social",
            "ok_smm": "social",
            "app": "social",
            "static": "social",

            # Partner/Promo
            "smartbanner": "partner",
            "partner": "partner",
            "promo_sber": "partner",
            "promo_sbol": "partner",
            "landing": "partner",
            "landing_interests": "partner",
            "link": "partner",
            "web_polka": "partner",
            "main_polka": "partner",
            "tablet": "partner",
            "desktop": "partner",
            "Sbol_catalog": "partner",
            "catalogue": "partner",
            "dom_click": "partner",

            # QR, rare
            "qr": "qr",
            "qrcodevideo": "qr"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column + "_mapped"] = X[self.column].apply(
            lambda x: self.mapping.get(str(x).lower(), "other") if isinstance(x, str) else "other")
        return X


class TopEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, top_n=10, is_hash=False):
        self.column = column
        self.top_n = top_n
        self.is_hash = is_hash
        self.top_categories_ = None

    def fit(self, X, y=None):
        self.top_categories_ = X[self.column].value_counts().head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        if self.is_hash:
            mapping = {cat: idx for idx, cat in enumerate(self.top_categories_)}
            mapping["other"] = self.top_n
            X[self.column + "_encoded"] = X[self.column].apply(lambda x: mapping.get(x, self.top_n))
        else:
            top_set = set(self.top_categories_)
            X[self.column + "_encoded"] = X[self.column].apply(lambda x: x if x in top_set else "other")
        return X


class DatetimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col, time_col):
        self.date_col = date_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["datetime"] = pd.to_datetime(df[self.date_col] + " " + df[self.time_col])

        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["hour"] = df["datetime"].dt.hour
        df["month"] = df["datetime"].dt.month

        df["season"] = df["month"].apply(lambda m: (
            "winter" if m in [12, 1, 2] else
            "spring" if m in [3, 4, 5] else
            "summer" if m in [6, 7, 8] else "autumn"
        ))

        return df.drop(columns=["datetime", "month", "hour"])


class FillNAAndStr(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna('other').astype(str)
        return X


class DropOriginalColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


custom_steps = Pipeline(steps=[
    ('map_medium', MapColumnTransformer(column='utm_medium')),
    ('encode_campaign', TopEncoderTransformer(column='utm_campaign', top_n=4, is_hash=True)),
    ('encode_source', TopEncoderTransformer(column='utm_source', top_n=7, is_hash=True)),
    ('encode_keyword', TopEncoderTransformer(column='utm_keyword', top_n=2, is_hash=True)),
    ('encode_device_os', TopEncoderTransformer(column='device_os', top_n=3)),
    ('encode_device_brand', TopEncoderTransformer(column='device_brand', top_n=4)),
    ('encode_device_browser', TopEncoderTransformer(column='device_browser', top_n=3)),
    ('encode_device_model', TopEncoderTransformer(column='device_model', top_n=1, is_hash=True)),
    ('encode_device_category', TopEncoderTransformer(column='device_category', top_n=1)),
    ('encode_geo_city', TopEncoderTransformer(column='geo_city', top_n=2)),
    ('encode_geo_country', TopEncoderTransformer(column='geo_country', top_n=1)),
    ('encode_visit_number', TopEncoderTransformer(column='visit_number', top_n=2)),
    ('datetime_features', DatetimeFeatureTransformer(date_col='visit_date', time_col='visit_time')),
    ('drop_orig', DropOriginalColumnsTransformer(columns_to_drop=[
        'utm_campaign', 'utm_source', 'utm_keyword', 'utm_medium', 'visit_number', 'visit_date', 'visit_time',
        'device_os', 'device_brand', 'device_browser', 'device_screen_resolution', 'device_category',
        'device_model',
        'geo_city', 'geo_country'
    ])),
])
