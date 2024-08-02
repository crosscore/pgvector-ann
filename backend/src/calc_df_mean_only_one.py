import pandas as pd
import os

# CATEGORY_NAMEの設定
CATEGORY_NAME = os.environ.get('CATEGORY_NAME', 'analytics_and_big_data')

# 入力ファイルのパス
before_search_file = f'../data/search_results_csv/before_search_{CATEGORY_NAME}.csv'
after_search_file = f'../data/search_results_csv/after_search_{CATEGORY_NAME}.csv'

# 出力ファイルのパス
before_search_output_file = f'../data/search_results_csv/before_search_{CATEGORY_NAME}_with_averages_one.csv'
after_search_output_file = f'../data/search_results_csv/after_search_{CATEGORY_NAME}_with_averages_one.csv'

def add_averages_to_csv(input_file, output_file):
    # CSVファイルの読み込み
    df = pd.read_csv(input_file, encoding="utf-8")

    # "index_type"カラム毎に平均値を計算
    avg_search_time = df.groupby('index_type')['search_time'].mean().round(4)
    avg_target_rank = df.groupby('index_type')['target_rank'].mean().round(1)
    avg_usage = df.groupby('index_type')['usage'].mean().round(1)

    # 固定値を取得（どの行の値でも良いので最初の値を使用）
    first_row = df.groupby('index_type').first()

    # 新しいデータフレームを作成
    result_df = first_row.copy()
    result_df['avg_search_time'] = avg_search_time
    result_df['avg_target_rank'] = avg_target_rank
    result_df['avg_usage'] = avg_usage

    # "index_type"をインデックスから列に戻す
    result_df.reset_index(inplace=True)

    # 必要なカラムのみを選択
    result_df = result_df[['index_type', 'hnsw_m', 'hnsw_ef_construction', 'hnsw_ef_search', 
                           'ivfflat_lists', 'ivfflat_probes', 'num_of_rows', 'usage', 
                           'avg_search_time', 'avg_target_rank', 'avg_usage']]

    # 新しいCSVファイルとして出力
    result_df.to_csv(output_file, index=False)

# 各ファイルに対して処理を実行
add_averages_to_csv(before_search_file, before_search_output_file)
add_averages_to_csv(after_search_file, after_search_output_file)
