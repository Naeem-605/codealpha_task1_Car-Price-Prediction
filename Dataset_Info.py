import pandas as pd
import datetime
df=pd.read_csv("E:\Code alpha Internship\Car price prediction\car data.csv")


# Injformation of Dataset
print("Dataset shape = ",df.shape)

print("Head of Datset: ",df.head())
# Columns check karne ke liye
print(df.columns)

# Sirf column names as list
print(list(df.columns))

# Aur agar detail dekhni ho
print(df.info())


