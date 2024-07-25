# rag-pgvector/backend/src/data_processing/drop_table.py
import os
import psycopg2
from psycopg2 import sql
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        return psycopg2.connect(
            dbname=os.getenv('PGVECTOR_DB_NAME'),
            user=os.getenv('PGVECTOR_DB_USER'),
            password=os.getenv('PGVECTOR_DB_PASSWORD'),
            host=os.getenv('PGVECTOR_DB_HOST'),
            port=os.getenv('PGVECTOR_DB_PORT')
        )
    except psycopg2.Error as e:
        logger.error(f"データベースへの接続に失敗しました: {e}")
        raise

def print_table_info(cursor):
    cursor.execute("SELECT COUNT(*) FROM document_vectors")
    row_count = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM document_vectors LIMIT 0")
    col_count = len(cursor.description)

    logger.info("\n------ テーブル情報 ------")
    logger.info(f"行数: {row_count}")
    logger.info(f"列数: {col_count}")
    logger.info("\n列の情報:")
    for col in cursor.description:
        logger.info(f"  - {col.name}: {col.type_code}")

def get_table_data(cursor):
    try:
        cursor.execute("SELECT COUNT(*) FROM document_vectors")
        row_count = cursor.fetchone()[0]
        logger.info(f"テーブルからデータを正常に読み込みました。行数: {row_count}")
        return True
    except psycopg2.Error as e:
        logger.error(f"テーブルデータの読み込み中にエラーが発生しました: {e}")
        return False

def drop_table(cursor):
    drop_table_query = sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier('document_vectors'))

    try:
        cursor.execute(drop_table_query)
        logger.info("テーブルが正常に削除されました。")
    except psycopg2.Error as e:
        logger.error(f"テーブル削除中にエラーが発生しました: {e}")

def main():
    logger.info("テーブル削除プロセスを開始します。")
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            if get_table_data(cursor):
                print_table_info(cursor)
                drop_table(cursor)
            else:
                logger.warning("テーブルが存在しないか、データの取得に失敗しました。")

        conn.commit()
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")
        conn.rollback()
    finally:
        conn.close()

    logger.info("テーブル削除プロセスが完了しました。")

if __name__ == "__main__":
    main()
