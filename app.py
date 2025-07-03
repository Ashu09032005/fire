import pandas as pd

chunksize = 100_000
df_list = []

for chunk in pd.read_csv("fire.csv", chunksize=chunksize):
    # Skip volcano rows
    chunk = chunk[chunk["type"] != 1]

    # Format time
    chunk['acq_time'] = chunk['acq_time'].astype(str).str.zfill(4)

    # Combine to datetime
    chunk['acq_datetime'] = pd.to_datetime(
        chunk['acq_date'].astype(str) + chunk['acq_time'],
        format='%Y-%m-%d%H%M',
        errors='coerce'
    )

    df_list.append(chunk)

# Merge all chunks
df = pd.concat(df_list, ignore_index=True)
df.drop(['acq_date', 'acq_time'], axis=1, inplace=True)

# Create future label
future_df = df[['acq_datetime', 'latitude', 'longitude']].copy()
future_df['acq_datetime'] = future_df['acq_datetime'] - pd.Timedelta(hours=2)
future_df.rename(columns={'latitude': 'lat_future', 'longitude': 'lon_future'}, inplace=True)

# Merge past with future
df = df.merge(future_df, on='acq_datetime', how='inner')

# Save
df.to_csv("fire_processed.csv", index=False)
print("✅ Processed file saved as fire_processed.csv")
