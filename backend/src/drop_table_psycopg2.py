import os
import psycopg2
from psycopg2 import sql
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv('PGVECTOR_DB_NAME'),
        user=os.getenv('PGVECTOR_DB_USER'),
        password=os.getenv('PGVECTOR_DB_PASSWORD'),
        host=os.getenv('PGVECTOR_DB_HOST'),
        port=os.getenv('PGVECTOR_DB_PORT')
    )

def get_all_public_tables(cursor):
    cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    """)
    return [row[0] for row in cursor.fetchall()]

def drop_all_tables(cursor, tables):
    for table in tables:
        drop_table_query = sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table))
        cursor.execute(drop_table_query)
        logger.info(f"テーブル {table} を削除しました。")

def main():
    logger.info("全ての公開テーブルの削除プロセスを開始します。")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                tables = get_all_public_tables(cursor)
                if not tables:
                    logger.info("削除すべき公開テーブルがありません。")
                else:
                    logger.info(f"削除対象のテーブル: {', '.join(tables)}")
                    drop_all_tables(cursor, tables)
                    conn.commit()
                    logger.info(f"合計 {len(tables)} 個のテーブルを削除しました。")
    except psycopg2.Error as e:
        logger.error(f"データベース操作中にエラーが発生しました: {e}")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")

    logger.info("テーブル削除プロセスが完了しました。")

if __name__ == "__main__":
    main()
