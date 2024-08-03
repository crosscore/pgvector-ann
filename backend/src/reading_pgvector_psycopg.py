import os
import logging
import psycopg
from psycopg.rows import dict_row
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
    return psycopg.connect(
        **db_params,
        options="-c timezone=Asia/Tokyo",
        row_factory=dict_row
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
    cursor.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = %s;
    """, (sanitized_name,))
    return cursor.fetchall()

def get_index_info(cursor, table_name):
    sanitized_name = sanitize_table_name(table_name)
    cursor.execute("""
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
    cursor.execute(f"SELECT * FROM {sanitized_name} LIMIT 1;")
    rows = cursor.fetchall()
    if rows:
        logger.info("------ Sample Data ------")
        for row in rows:
            formatted_row = {}
            for key, value in row.items():
                if key == 'chunk_vector':
                    logger.info(f"Type of chunk_vector: {type(value)}")
                    formatted_row[key] = value[:27] + " ...]" if len(value) > 20 else value
                else:
                    formatted_row[key] = value
            logger.info(formatted_row)
    else:
        logger.warning(f"No sample data found for {table_name}.")

def get_record_count(cursor, table_name):
    sanitized_name = sanitize_table_name(table_name)
    cursor.execute(f"SELECT COUNT(*) FROM {sanitized_name};")
    return cursor.fetchone()['count']

def get_all_tables(cursor):
    cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public';
    """)
    return [row['table_name'] for row in cursor.fetchall()]

def get_table_summary(cursor):
    tables = get_all_tables(cursor)
    summary = []
    for table in tables:
        record_count = get_record_count(cursor, table)
        summary.append((table, record_count))
    return summary

def print_table_summary(summary):
    logger.info("------ Table Summary ------")
    for table_name, record_count in summary:
        logger.info(f"Table: {table_name}, Records: {record_count}")

def main():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        tables = get_all_tables(cursor)

        if not tables:
            logger.warning("No tables found in the database.")
        else:
            logger.info(f"Found {len(tables)} tables in the database.")

        for table_name in tables:
            logger.info(f"\n===== Table: {table_name} =====")

            table_structure = get_table_structure(cursor, table_name)
            logger.info("------ Table Structure ------")
            if table_structure:
                for column in table_structure:
                    logger.info(f"  - {column['column_name']}: {column['data_type']}")
            else:
                logger.warning(f"No table structure information found for {table_name}.")

            logger.info("\n------ Index Information ------")
            index_info = get_index_info(cursor, table_name)
            if index_info:
                for index in index_info:
                    logger.info(f"Index Name: {index['index_name']}")
                    logger.info(f"  Column: {index['column_name']}")
                    logger.info(f"  Type: {index['index_type']}")
                    logger.info(f"  Definition: {index['index_definition']}")
                    logger.info("  ---")
            else:
                logger.warning(f"No index information found for {table_name}.")

            log_sample_data(cursor, table_name)

            record_count = get_record_count(cursor, table_name)
            logger.info(f"\n------ Record Count ------")
            logger.info(f"Total record count for {table_name} table: {record_count}")

        table_summary = get_table_summary(cursor)
        print_table_summary(table_summary)

        logger.info("\nDatabase reading completed.")
    except psycopg.Error as e:
        logger.error(f"Database error occurred: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
