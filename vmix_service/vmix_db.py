import logging.config
from os import path

import mysql.connector
import mysql.connector
from mysql.connector import errorcode

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logger.config')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('app_logger')

config = {
    'user': 'root',
    'password': 'password',
    'host': '127.0.0.1',
    'database': 'vmix_video',
    'raise_on_warnings': True
}


def record_file_info(file_info):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        insert_job_record = ("INSERT INTO live_file(file_name, size, from_client, c_time, m_time) "
                             "VALUES (%s, %s, %s, %s, %s)")
        job_data = (file_info.file_name, file_info.size, file_info.from_client, file_info.c_time, file_info.m_time)
        cursor.execute(insert_job_record, job_data)
        file_id = cursor.lastrowid
        logger.info("live file record is inserted. %d %s" % (file_id, file_info.file_name))
        cnx.commit()
    except Exception as e:
        logger.error(e)
    finally:
        close_connect(cnx)


def record_upload_file_status(status, file_name, from_client, c_time):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        update_job_status = ("UPDATE live_file set status=%s "
                             "where file_name=%s and from_client=%s")
        status_param = (status, file_name, from_client)
        logger.info(update_job_status)
        result = cursor.execute(update_job_status, status_param)
        cnx.commit()
    except Exception as err:
        logger.error(err)
        raise err
    finally:
        close_connect(cnx)


def record_upload_job(by_client, files_amount):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        insert_job_record = ("INSERT INTO upload_job(from_client, files_amount) "
                             "VALUES (%s, %s)")
        job_data = (by_client, files_amount)
        cursor.execute(insert_job_record, job_data)
        job_id = cursor.lastrowid
        logger.info("upload job id %d" % job_id)
        cnx.commit()
    except Exception as err:
        logger.error(err)
        raise err
    finally:
        close_connect(cnx)


def record_upload_job_status(job_id, status):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        update_job_status = "UPDATE upload_job set status=%d where job_id= " % (status, job_id)
        logger.info(update_job_status)
        cursor.execute(update_job_status)
        cnx.commit()
    except Exception as err:
        logger.error(err)
        raise err
    finally:
        close_connect(cnx)


def close_connect(cnx):
    try:
        cnx.close()
    except Exception as e:
        print(e)


def get_latest_upload_job(by_client):
    try:
        cnx = mysql.connector.connect(**config)
        cursor = cnx.cursor()
        query = "SELECT id from upload_job WHERE from_client=\"%s\" ORDER BY start_time desc limit 1" % by_client
        cursor.execute(query)
        record = cursor.fetchone()
        logger.info("job id %d " % record[0])
        return record[0]
    except Exception as err:
        logger.error(err)
        raise err
    finally:
        try:
            cnx.close()
        except Exception as e:
            print(e)
