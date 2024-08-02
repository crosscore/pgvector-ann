import pandas as pd
import glob
import os

# df = pd.read_csv("../data/csv/all/all.csv", encoding="utf-8")
# print(df)

# df['file_name'] = 'securing-iot-with-aws.pdf'
# df.to_csv("../data/csv/iot/securing-iot-with-aws_modified.csv", index=False, encoding="utf-8")

# file_nameに"securing-iot-with-aws"の文字列が含まれる行のfile_nameカラムの値を変更
# df.loc[df['file_name'].str.contains("securing-iot-with-aws"), 'file_name'] = 'securing-iot-with-aws.pdf'


# Get the list of CSV files in the specified directory
# csv_files = glob.glob("../../data/search_csv/*.csv")

# if not csv_files:
#     raise ValueError("No CSV files found in the specified directory.")

# for csv_file in csv_files:
#     # Read the CSV file
#     df = pd.read_csv(csv_file, encoding="utf-8")
    
#     # Drop the last row
#     df = df[:-1]
    
#     # Print the updated DataFrame (optional)
#     print(f"Updated DataFrame for {csv_file}:\n", df)
    
#     # Save the updated DataFrame back to the CSV file
#     df.to_csv(csv_file, index=False, encoding="utf-8")


csv_files = glob.glob("../../data/search_results_csv/*.csv")
print(csv_files)

for file in csv_files:
    df = pd.read_csv(file)
    df = df[df['index_type'] != 'hnsw']
    df.to_csv(file, index=False)
