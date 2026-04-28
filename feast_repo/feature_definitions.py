from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

cc_num = Entity(
    name="cc_num",
    description="Credit card number — primary entity key",
)

fraud_source = FileSource(
    path="/app/training/features.parquet",
    timestamp_field="event_timestamp",
)

fraud_features = FeatureView(
    name="fraud_features",
    entities=[cc_num],
    ttl=timedelta(days=365),
    schema=[
        Field(name="amt",        dtype=Float64),
        Field(name="lat",        dtype=Float64),
        Field(name="long",       dtype=Float64),
        Field(name="city_pop",   dtype=Int64),
        Field(name="unix_time",  dtype=Int64),
        Field(name="merch_lat",  dtype=Float64),
        Field(name="merch_long", dtype=Float64),
    ],
    source=fraud_source,
    online=True,
)
