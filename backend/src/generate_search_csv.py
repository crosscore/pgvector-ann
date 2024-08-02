import pandas as pd
import os
import glob
import random
import re
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

CATEGORY_NAME = os.environ.get('CATEGORY_NAME', 'analytics_and_big_data')

input_directory = f'../data/csv/{CATEGORY_NAME}'
output_file = f'../data/search_csv/search_{CATEGORY_NAME}.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def is_valid_sentence(sentence):
    # 空白を除去
    cleaned_sentence = sentence.strip()

    # 連続した2つ以上のピリオドを含む行を除外
    if re.search(r'\.{2,}', cleaned_sentence):
        logging.debug(f"Excluded due to consecutive periods: {cleaned_sentence}")
        return False

    # 長さが20文字未満の場合を除外
    if len(cleaned_sentence) < 20:
        logging.debug(f"Excluded due to short length: {cleaned_sentence}")
        return False

    logging.debug(f"Valid sentence: {cleaned_sentence}")
    return True

def select_valid_sentence(text):
    sentences = text.split('\n')
    logging.debug(f"Split sentences: {sentences}")

    valid_sentences = [sent.strip() for sent in sentences if is_valid_sentence(sent.strip())]

    if valid_sentences:
        selected = random.choice(valid_sentences)
        logging.info(f"Selected sentence: {selected}")
        return selected
    else:
        logging.warning(f"No valid sentence found in: {text[:100]}...")
        return ""

# 結果を格納するDataFrame
result_df = pd.DataFrame(columns=['file_name', 'document_page', 'search_text'])

# 指定されたディレクトリ内のすべてのCSVファイルを処理
for csv_file in glob.glob(os.path.join(input_directory, '*.csv')):
    logging.info(f"Processing file: {csv_file}")
    df = pd.read_csv(csv_file)

    # ファイルの行数が10未満の場合、全ての行を使用。それ以外の場合は10行をランダムに選択
    sample_df = df if len(df) <= 10 else df.sample(n=10)

    # 必要なカラムだけを選択し、search_textを生成
    sample_df = sample_df[['file_name', 'document_page', 'chunk_text']]
    sample_df['search_text'] = sample_df['chunk_text'].apply(select_valid_sentence)

    # 空の search_text を持つ行を削除
    sample_df = sample_df[sample_df['search_text'] != ""]

    # chunk_text列を削除
    sample_df = sample_df.drop('chunk_text', axis=1)

    # 結果をresult_dfに追加
    result_df = pd.concat([result_df, sample_df], ignore_index=True)

# 重複を削除（同じfile_nameとdocument_pageの組み合わせで最初の行を保持）
result_df = result_df.drop_duplicates(subset=['file_name', 'document_page'], keep='first')

# 列の順序を整理
result_df = result_df[['file_name', 'document_page', 'search_text']]

# 最終チェック：連続したピリオドを含む行がないか確認
final_check = result_df[result_df['search_text'].str.contains(r'\.{2,}', regex=True)]
if not final_check.empty:
    logging.error("Found rows with consecutive periods after all processing:")
    for _, row in final_check.iterrows():
        logging.error(f"File: {row['file_name']}, Page: {row['document_page']}, Text: {row['search_text']}")
    # 問題のある行を削除
    result_df = result_df[~result_df.index.isin(final_check.index)]
    logging.info("Removed problematic rows.")

result_df.to_csv(output_file, index=False)

logging.info(f"Search CSV file has been generated: {output_file}")
logging.info(f"Total rows in search CSV: {len(result_df)}")