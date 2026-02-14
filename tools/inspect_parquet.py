import pandas as pd
p = 'data_files/espn_pga_2018_2025.parquet'
df = pd.read_parquet(p)
print('rows,cols', df.shape)
print('cols:', df.columns.tolist())
print('\ncols types:\n')
print(df.dtypes)
print('\nsample rows:\n')
print(df.head(3).to_dict(orient='records'))
