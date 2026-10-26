import pandas as pd
import io

print("=" * 60)
print("PANDAS READ_CSV ADVANCED PARAMETERS DEMO")
print("=" * 60)

# 1. Simulating a dirty custom-delimited dataset with comments & missing values
dirty_csv_stream = """# Database Export Metadata: Version 1.2
# Generated on: 2026-10-19
id|cgpa|iq|placement|status
101|7.8|105|1|active
102|6.5|95|0|inactive
103|?|110|1|pending
104|8.9|120|1|active|extra_bad_field
105|5.2|88|0|active
"""

# Read using pipe delimiter, skipping comments, converting '?' to NaN, and skipping bad lines
df = pd.read_csv(
    io.StringIO(dirty_csv_stream),
    sep="|",
    comment="#",
    na_values=["?", "pending"],
    on_bad_lines="skip",
    usecols=["id", "cgpa", "iq", "placement"],
    dtype={"placement": "int8"}
)

print("Parsed DataFrame:")
print(df)
print("\nData Types:")
print(df.dtypes)
print("\nMissing values detected:")
print(df.isna().sum())