import pandas as pd
import numpy as np
df=pd.read_csv("../ratings.dat",sep="::",header=None,engine="python")
df.drop(df.columns[[3]], axis=1,inplace=True)
df.columns=['user_id','movies_id','rating']

no_movies=max(df["movies_id"])
no_users=max(df["user_id"])


y=np.zeros((no_movies,no_users))
for index, row in df.iterrows():
	y[row['movies_id']-1][row['user_id']-1]=row['rating'] # Because numpy arrays are 1-indexed
	#i'th row and j'th column of y represents rating of i+1'th movie by j+1'th user

h = np.array(y,dtype=float)

#  Write code to generate R matrix of zeroes and ones
R = y>0.5
R[R == True] = 1
R[R == False] = 0
# To be consistent with our Q matrix
R = R.astype(np.float64, copy=False)


import pickle
with open("data_y","wb") as f:
	pickle.dump(y,f)

with open("data_R","wb") as f:
	pickle.dump(R,f)
