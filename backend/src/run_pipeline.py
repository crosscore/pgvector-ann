import subprocess
import time
import logging
from datetime import datetime
import os
import csv
import pandas as pd
from config import INDEX_TYPE, PDF_INPUT_DIR, CSV_OUTPUT_DIR

# Ensure the necessary directories exist
log_dir = "/app/data/log"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(PDF_INPUT_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# Set up logging
log_file = f"{log_dir}/run_pipeline.log"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(message)s')

csv_output_file = f"{log_dir}/run_pipeline.csv"

def count_csv_rows(directory):
    total_rows = 0
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            with open(os.path.join(directory, filename), 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = sum(1 for row in reader) - 1  # Subtract 1 to exclude header
                total_rows += rows
    return total_rows

def append_to_csv(filename, index_type, num_of_rows, execution_time):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(csv_output_file):
        df = pd.read_csv(csv_output_file)
        new_index = df['index'].max() + 1 if not df.empty else 0
    else:
        df = pd.DataFrame(columns=['index', 'filename', 'index_type', 'num_of_rows', 'execution_time', 'timestamp'])
        new_index = 0

    new_row = pd.DataFrame({
        'index': [new_index],
        'filename': [filename],
        'index_type': [index_type],
        'num_of_rows': [num_of_rows],
        'execution_time': [round(execution_time, 2)],  # Round to 2 decimal places
        'timestamp': [timestamp]
    })

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_output_file, index=False)
    logging.info(f"Appended execution data to {csv_output_file}")

def run_script(script_name):
    start_time = time.time()
    logging.info(f"Starting execution of {script_name}")

    try:
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        end_time = time.time()
        execution_time = end_time - start_time

        # Log the summary
        logging.info(f"{script_name} executed successfully:")
        logging.info(f"  - Execution time: {execution_time:.2f} seconds")  # Format to 2 decimal places in log
        if script_name == 'csv_to_pgvector.py':
            row_count = count_csv_rows(CSV_OUTPUT_DIR)
            logging.info(f"  - Index type: {INDEX_TYPE.upper()}")
            logging.info(f"  - Rows inserted: {row_count}")
            append_to_csv(script_name, INDEX_TYPE.upper(), row_count, execution_time)
        elif script_name == 'vectorizer.py':
            pdf_count = len([f for f in os.listdir(PDF_INPUT_DIR) if f.endswith('.pdf')])
            csv_count = len([f for f in os.listdir(CSV_OUTPUT_DIR) if f.endswith('.csv')])
            logging.info(f"  - PDFs processed: {pdf_count}")
            logging.info(f"  - CSV files generated: {csv_count}")
            append_to_csv(script_name, 'N/A', csv_count, execution_time)

    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing {script_name}: {e}")
        logging.error(f"Error output: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error occurred while running {script_name}: {e}")

def main():
    logging.info("Starting pipeline execution")
    logging.info(f"Using index type: {INDEX_TYPE.upper()}")

    run_script('vectorizer.py')
    run_script('csv_to_pgvector.py')

    logging.info("Pipeline execution completed")

if __name__ == "__main__":
    main()
