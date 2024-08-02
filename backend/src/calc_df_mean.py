import pandas as pd
import os

# CATEGORY_NAMEの設定
CATEGORY_NAME = os.environ.get('CATEGORY_NAME', 'analytics_and_big_data')

# 入力ファイルのパス
before_search_file = f'../data/search_results_csv/before_search_{CATEGORY_NAME}.csv'
after_search_file = f'../data/search_results_csv/after_search_{CATEGORY_NAME}.csv'

# 出力ファイルのパス
before_search_output_file = f'../data/search_results_csv/before_search_{CATEGORY_NAME}_with_averages.csv'
after_search_output_file = f'../data/search_results_csv/after_search_{CATEGORY_NAME}_with_averages.csv'

def add_averages_to_csv(input_file, output_file):
    # CSVファイルの読み込み (エンコーディング指定)
    df = pd.read_csv(input_file, encoding='utf-8')

    # "index_type"カラム毎に平均値を計算
    avg_search_time = df.groupby('index_type')['search_time'].transform('mean').round(4)
    avg_target_rank = df.groupby('index_type')['target_rank'].transform('mean').round(1)
    avg_usage = df.groupby('index_type')['usage'].transform('mean').round(1)

    # 新しいカラムとして追加
    df['avg_search_time'] = avg_search_time
    df['avg_target_rank'] = avg_target_rank
    df['avg_usage'] = avg_usage

    # 新しいCSVファイルとして出力 (BOM付きUTF-8)
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as f:
        df.to_csv(f, index=False)

# 各ファイルに対して処理を実行
add_averages_to_csv(before_search_file, before_search_output_file)
add_averages_to_csv(after_search_file, after_search_output_file)
