import os.path as osp
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df = pd.read_csv('../../data/yoochoose_data/yoochoose-clicks.dat', header=None)
df.columns=['session_id', 'timestamp', 'item_id', 'category']

buy_df = pd.read_csv('../../data/yoochoose_data/yoochoose-buys.dat', header=None)
buy_df.columns=['session_id','timestamp','item_id','price','quantity']

item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)
print(df.head())

# randomly subsample for easier demonstration because data is quite large
sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
print(df.nunique())

