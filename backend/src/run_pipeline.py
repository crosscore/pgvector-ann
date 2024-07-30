# pgvector-ann/backend/src/run_pipeline.py
import subprocess
import time
import logging
from datetime import datetime
import os
import csv
import pandas as pd
from config import INDEX_TYPE, PDF_INPUT_DIR, CSV_OUTPUT_DIR, PIPELINE_EXECUTION_MODE

log_dir = "/app/data/log"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(PDF_INPUT_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

vectorizer_log = f"{log_dir}/vectorizer.log"
csv_to_pgvector_log = f"{log_dir}/csv_to_pgvector.log"
run_pipeline_log = f"{log_dir}/run_pipeline.log"

logging.basicConfig(filename=run_pipeline_log, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

csv_output_file = f"{log_dir}/run_pipeline.csv"
ENABLE_ALL_CSV = os.getenv("ENABLE_ALL_CSV", "false").lower() == "true"

def count_csv_rows(directory):
    if ENABLE_ALL_CSV:
        all_csv_path = "/app/data/csv/all/all.csv"
        if os.path.exists(all_csv_path):
            with open(all_csv_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                return sum(1 for row in reader) - 1  # Subtract 1 to exclude header
        else:
            return 0
    else:
        total_rows = 0
        for root, _, files in os.walk(directory):
            if "all" in root.split(os.path.sep):
                continue
            for filename in files:
                if filename.endswith('.csv'):
                    with open(os.path.join(root, filename), 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        rows = sum(1 for row in reader) - 1  # Subtract 1 to exclude header
                        total_rows += rows
        return total_rows

def get_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f}MB"

def append_to_csv(filename, index_type, num_of_rows, execution_time):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({
        'index': [0],  # 仮の値を設定
        'filename': [filename],
        'index_type': [index_type],
        'num_of_rows': [num_of_rows],
        'execution_time': [round(execution_time, 2)],
        'timestamp': [timestamp]
    })

    if os.path.exists(csv_output_file):
        df = pd.read_csv(csv_output_file)
        new_row['index'] = df['index'].max() + 1 if not df.empty else 0
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        new_row['index'] = 0
        df = new_row

    df.to_csv(csv_output_file, index=False)
    logging.info(f"Appended execution data to {csv_output_file}")

def run_script(script_name):
    start_time = time.time()
    logger.info(f"Starting execution of {script_name}")

    try:
        # スクリプトを実行
        process = subprocess.Popen(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"{script_name} failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return

        end_time = time.time()
        execution_time = end_time - start_time

        # Log the summary
        logger.info(f"{script_name} executed successfully:")
        logger.info(f"  - Execution time: {execution_time:.2f} seconds")
        if script_name == 'csv_to_pgvector.py':
            row_count = count_csv_rows(CSV_OUTPUT_DIR)
            logger.info(f"  - Index type: {INDEX_TYPE.upper()}")
            logger.info(f"  - Rows inserted: {row_count}")
            logger.info(f"  - ENABLE_ALL_CSV: {ENABLE_ALL_CSV}")
            append_to_csv(script_name, INDEX_TYPE.upper(), row_count, execution_time)
        elif script_name == 'vectorizer.py':
            pdf_count = sum([len(files) for _, _, files in os.walk(PDF_INPUT_DIR) if any(f.endswith('.pdf') for f in files)])
            csv_count = sum([len(files) for _, _, files in os.walk(CSV_OUTPUT_DIR) if any(f.endswith('.csv') for f in files)])
            logger.info(f"  - PDFs processed: {pdf_count}")
            logger.info(f"  - CSV files generated: {csv_count}")
            append_to_csv(script_name, INDEX_TYPE.upper(), csv_count, execution_time)

    except Exception as e:
        logger.error(f"Unexpected error occurred while running {script_name}: {e}")

def combine_logs():
    log_files = [vectorizer_log, csv_to_pgvector_log, run_pipeline_log]
    combined_log = f"{log_dir}/combined_pipeline.log"

    with open(combined_log, 'w') as outfile:
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, 'r') as infile:
                    outfile.write(infile.read())

    # ログファイルをソート
    with open(combined_log, 'r') as f:
        sorted_logs = sorted(f.readlines(), key=lambda x: x.split(' - ')[0])

    with open(combined_log, 'w') as f:
        f.writelines(sorted_logs)

    logger.info(f"Combined and sorted log file created: {combined_log}")

def main():
    logger.info("Starting pipeline execution")
    logger.info(f"Using index type: {INDEX_TYPE.upper()}")
    logger.info(f"ENABLE_ALL_CSV: {ENABLE_ALL_CSV}")
    logger.info(f"PIPELINE_EXECUTION_MODE: {PIPELINE_EXECUTION_MODE}")

    if PIPELINE_EXECUTION_MODE == "BOTH":
        run_script('vectorizer.py')
        run_script('csv_to_pgvector.py')
    elif PIPELINE_EXECUTION_MODE == "csv_to_pgvector":
        run_script('csv_to_pgvector.py')
    else:
        logger.error(f"Invalid PIPELINE_EXECUTION_MODE: {PIPELINE_EXECUTION_MODE}")
        return

    logger.info("Pipeline execution completed")
    combine_logs()

if __name__ == "__main__":
    main()
