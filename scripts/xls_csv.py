import pandas as pd
data_xls = pd.read_excel('../datasets_raw/JETAIRWAYS-I.xlsx', 'Sheet1', index_col=None)
data_xls.to_csv('../datasets_raw/JETAIRWAYS-I.csv', encoding='utf-8')