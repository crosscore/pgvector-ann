# pgvector-ann/backend/src/reading_pgvector_psycopg2.py
import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    db_params = {
        'dbname': os.getenv('PGVECTOR_DB_NAME'),
        'user': os.getenv('PGVECTOR_DB_USER'),
        'password': os.getenv('PGVECTOR_DB_PASSWORD'),
        'host': os.getenv('PGVECTOR_DB_HOST'),
        'port': os.getenv('PGVECTOR_DB_PORT')
    }
    logger.info(f"Attempting to connect to database with params: {db_params}")
    return psycopg2.connect(
        **db_params,
        options="-c timezone=Asia/Tokyo",
    )

def sanitize_table_name(name):
    # Remove any character that isn't alphanumeric or underscore
    sanitized = re.sub(r'\W+', '_', name)
    # Ensure the name starts with a letter
    if not sanitized[0].isalpha():
        sanitized = "t_" + sanitized
    return sanitized.lower()

def get_table_structure(cursor, table_name):
    sanitized_name = sanitize_table_name(table_name)
    cursor.execute(f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = %s;
    """, (sanitized_name,))
    return cursor.fetchall()

def get_index_info(cursor, table_name):
    sanitized_name = sanitize_table_name(table_name)
    cursor.execute(f"""
    SELECT
        i.relname AS index_name,
        a.attname AS column_name,
        am.amname AS index_type,
        pg_get_indexdef(i.oid) AS index_definition
    FROM
        pg_index ix
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_am am ON i.relam = am.oid
        LEFT JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
    WHERE
        t.relname = %s
    ORDER BY
        i.relname, a.attnum;
    """, (sanitized_name,))
    return cursor.fetchall()

def log_sample_data(cursor, table_name):
    sanitized_name = sanitize_table_name(table_name)
    logger.info(f"\n------ サンプルデータ (最初の1行) from {sanitized_name} ------")
    try:
        cursor.execute(f"""
        SELECT
            file_name, document_page, chunk_no, chunk_text, model,
            prompt_tokens, total_tokens, created_date_time,
            chunk_vector, business_category
        FROM {sanitized_name}
        LIMIT 1
        """)

        sample = cursor.fetchone()

        if sample:
            for key, value in sample.items():
                if key == 'created_date_time':
                    if value.tzinfo is None:
                        value = value.replace(tzinfo=timezone.utc)
                    jst_time = value.astimezone(ZoneInfo("Asia/Tokyo"))
                    logger.info(f"{key}: {type(value).__name__} - {jst_time}")
                elif key == 'chunk_vector':
                    vector_str = value.strip('[]')
                    vector_list = [float(x) for x in vector_str.split(',')]
                    vector_sample = vector_list[:5]
                    logger.info(f"{key}: vector({len(vector_list)}) - {vector_sample} (First 5 elements)")
                    logger.info(f"Vector min: {min(vector_list)}, max: {max(vector_list)}, avg: {sum(vector_list)/len(vector_list):.4f}")
                else:
                    logger.info(f"{key}: {type(value).__name__} - {value}")
        else:
            logger.warning(f"No data found in the {sanitized_name} table.")
    except psycopg2.Error as e:
        logger.error(f"Error querying sample data from {sanitized_name}: {e}")

def get_record_count(cursor, table_name):
    sanitized_name = sanitize_table_name(table_name)
    cursor.execute(f"SELECT COUNT(*) FROM {sanitized_name}")
    return cursor.fetchone()['count']

def get_all_tables(cursor):
    cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    """)
    return [row['table_name'] for row in cursor.fetchall()]

def get_table_summary(cursor):
    tables = get_all_tables(cursor)
    summary = []
    for table_name in tables:
        sanitized_name = sanitize_table_name(table_name)
        cursor.execute(f"SELECT COUNT(*) FROM {sanitized_name}")
        count = cursor.fetchone()['count']
        summary.append((table_name, count))
    return summary

def print_table_summary(summary):
    logger.info("\n===== テーブル一覧とレコード数 =====")
    for table_name, count in summary:
        logger.info(f"{table_name}: {count} レコード")

def main():
    conn = None
    cursor = None
    try:
        logger.info("データベースからの読み取りを開始します。")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        tables = get_all_tables(cursor)

        if not tables:
            logger.warning("No tables found in the database.")
        else:
            logger.info(f"Found {len(tables)} tables in the database.")

        for table_name in tables:
            logger.info(f"\n===== テーブル: {table_name} =====")

            table_structure = get_table_structure(cursor, table_name)
            logger.info("------ テーブル構造 ------")
            if table_structure:
                for column in table_structure:
                    logger.info(f"  - {column['column_name']}: {column['data_type']}")
            else:
                logger.warning(f"No table structure information found for {table_name}.")

            logger.info("\n------ インデックス情報 ------")
            index_info = get_index_info(cursor, table_name)
            if index_info:
                for index in index_info:
                    logger.info(f"インデックス名: {index['index_name']}")
                    logger.info(f"  カラム: {index['column_name']}")
                    logger.info(f"  タイプ: {index['index_type']}")
                    logger.info(f"  定義: {index['index_definition']}")
                    logger.info("  ---")
            else:
                logger.warning(f"No index information found for {table_name}.")

            log_sample_data(cursor, table_name)

            record_count = get_record_count(cursor, table_name)
            logger.info(f"\n------ レコード数 ------")
            logger.info(f"{table_name} テーブルの総レコード数: {record_count}")

        # 新しく追加した部分: テーブル一覧とレコード数の表示
        table_summary = get_table_summary(cursor)
        print_table_summary(table_summary)

        logger.info("\nデータベースの読み取りが完了しました。")
    except psycopg2.Error as e:
        logger.error(f"データベースエラーが発生しました: {e}")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
