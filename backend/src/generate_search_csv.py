# pgvector-ann/backend/src/generate_search_csv.py
import pandas as pd
import os
import glob
import random

CATEGORY_NAME = os.environ.get('CATEGORY_NAME', 'analytics_and_big_data')

input_directory = f'../data/csv/{CATEGORY_NAME}'
output_file = f'../data/csv/search/search_{CATEGORY_NAME}.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def extract_first_sentence(text):
    return text.split('\n')[0]

# 結果を格納するDataFrame
result_df = pd.DataFrame(columns=['file_name', 'document_page', 'search_text', 'chunk_text'])

# 指定されたディレクトリ内のすべてのCSVファイルを処理
for csv_file in glob.glob(os.path.join(input_directory, '*.csv')):
    df = pd.read_csv(csv_file)

    # ファイルの行数が10未満の場合、全ての行を使用
    if len(df) <= 10:
        sample_df = df
    else:
        # ランダムに10行を選択
        sample_df = df.sample(n=10)

    # 必要なカラムだけを選択し、search_textを生成
    sample_df = sample_df[['file_name', 'document_page', 'chunk_text']]
    sample_df['search_text'] = sample_df['chunk_text'].apply(extract_first_sentence)

    # 結果をresult_dfに追加
    result_df = pd.concat([result_df, sample_df], ignore_index=True)

# 重複を削除（同じfile_nameとdocument_pageの組み合わせで最初の行を保持）
result_df = result_df.drop_duplicates(subset=['file_name', 'document_page'], keep='first')

# 列の順序を整理
result_df = result_df[['file_name', 'document_page', 'search_text', 'chunk_text']]

result_df.to_csv(output_file, index=False)

print(f"Search CSV file has been generated: {output_file}")
print(f"Total rows in search CSV: {len(result_df)}")
