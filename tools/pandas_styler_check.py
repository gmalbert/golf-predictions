import pandas as pd
print('pandas', pd.__version__)
print([m for m in dir(pd.DataFrame().style) if 'hide' in m.lower() or 'index' in m.lower()])
