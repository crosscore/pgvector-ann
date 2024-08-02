import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(message)s')
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

def get_table_structure(cursor, table_name):
    cursor.execute(f"""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = %s;
    """, (table_name,))
    return cursor.fetchall()

def get_index_info(cursor, table_name):
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
    """, (table_name,))
    return cursor.fetchall()

def log_sample_data(cursor, table_name):
    logger.info(f"\n------ サンプルデータ (最初の1行) from {table_name} ------")
    cursor.execute(f"""
    SELECT
        file_name, document_page, chunk_no, chunk_text, model,
        prompt_tokens, total_tokens, created_date_time,
        chunk_vector
    FROM {table_name}
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
                vector_sample = vector_list[:2]
                logger.info(f"{key}: vector({len(vector_list)}) - {vector_sample} (First 2 elements)\n")
            else:
                logger.info(f"{key}: {type(value).__name__} - {value}")
    else:
        logger.warning(f"No data found in the {table_name} table.")

def get_record_count(cursor, table_name):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()['count']

def get_all_tables(cursor):
    cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name LIKE 'document_vectors%'
    """)
    return [row['table_name'] for row in cursor.fetchall()]

def main():
    conn = None
    cursor = None
    try:
        logger.info("データベースからの読み取りを開始します。")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        tables = get_all_tables(cursor)
        
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