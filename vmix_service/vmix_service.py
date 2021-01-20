from jproperties import Properties
import logging.config
import os
import time
from datetime import datetime
from ftplib import FTP
from os import path

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from flask import jsonify
from flask import make_response
from flask import request
from jproperties import Properties

import UploadJob
from UploadJob import LiveFile
from vmix_db import get_latest_upload_job
from vmix_db import record_file_info
from vmix_db import record_upload_file_status
from vmix_db import record_upload_job
from vmix_db import record_upload_job_status

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logger.config')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('app_logger')

app = Flask(__name__)
scheduler = BackgroundScheduler()

VIDEO_UPLOAD_JOB_ID = 'video_files_upload'
global SETTINGS
SETTINGS = {}


@app.route('/api/v1/')
def index():
    return "vmixVideoFileUploader"


@app.route('/api/v1/setting', methods=['POST'])
def set_settings():
    settings = request.get_json()
    save_setting_in_file(settings)

    reschedule_upload_video_files(int(SETTINGS['upload_hour']), int(SETTINGS['upload_minute']))

    return jsonify(settings), 200


@app.errorhandler(411)
def resource_not_found(error):
    return make_response(jsonify({'error': 'Missing field(s) in app-config.properties!'}), 400)


@app.route('/api/v1/setting', methods=['GET'])
def local_setting():
    return jsonify(get_local_setting()), 200


@app.route('/api/v1/status', methods=['GET'])
def status():
    files = list_recorded_video_files(SETTINGS['vmix_videos_dir'])
    files_numb = len(files)
    total_size = 0
    files_info = []
    for file in files:
        file_size = os.path.getsize(file)
        created_time = time.localtime(os.stat(file).st_ctime)
        modified_time = time.localtime(os.stat(file).st_mtime)
        ctime = time.strftime('%Y-%m-%d %H:%M:%S', created_time)
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', modified_time)
        total_size += file_size

        files_info.append((file, ctime, mtime, file_size))

    files_info = {
        'files': files_info,
        'total_numb': files_numb,
        'total_size': total_size
    }

    logger.info(files_info)
    return jsonify(files_info), 200


def get_local_setting():
    settings = {}
    configs = Properties()
    with open('app-config.properties', 'rb') as config_file:
        configs.load(config_file)

    items = configs.items()
    for item in items:
        settings[item[0]] = item[1].data
    return settings


def save_setting_in_file(settings):
    configs = Properties()
    for key, value in settings.items():
        configs[key] = '%s' % value
    with open("app-config.properties", "wb") as config_file:
        configs.store(config_file, encoding="utf-8")

    global SETTINGS
    SETTINGS = get_local_setting()


def list_recorded_video_files(file_dir):
    files = []
    # TODO：应该只返回已经完成的视频。（比如一秒内文件长度没有变化、或者文件时间戳为一小时前）
    for file in os.listdir(file_dir):
        files.append(os.path.join(file_dir, file))
    return files


def upload_files():
    ftp_server = SETTINGS['ftp_server']
    ftp_user = SETTINGS['ftp_user']
    ftp_pwd = SETTINGS['ftp_pwd']
    ftp_block_size = int(SETTINGS['ftp_block_size'])
    vmix_video_root_dir = SETTINGS['vmix_videos_dir']
    vmix_name = SETTINGS['vmix_name']

    files_to_upload = list_recorded_video_files(vmix_video_root_dir)
    files_amount = len(files_to_upload)
    if files_amount <= 0:
        logger.info("There is no video files from %s" % vmix_name)
        return

    upload_job_id = record_upload_job_start_in_db(files_to_upload, vmix_name)
    upload_counter = 0
    try:
        ftp = FTP(host=ftp_server, user=ftp_user, passwd=ftp_pwd)
        ftp_home_dir = ftp.pwd()
        logger.info('Current dir %s' % ftp_home_dir)

        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        upload_dir = "%s/%s" % (ftp_home_dir, time_str)
        logger.info("Create dir %s on FTP server." % upload_dir)
        ftp.mkd(upload_dir)
        ftp.cwd(upload_dir)

        for file_name in files_to_upload:
            file_size = path.getsize(file_name)
            upload_tracker = FtpUploadTracker(file_name, file_size, ftp_block_size)
            with open(file_name, 'rb') as file:
                base_name = os.path.basename(file.name)
                store_resp = ftp.storbinary('STOR %s' % base_name, file, ftp_block_size, upload_tracker.handle)
                logger.info(store_resp)
                upload_counter += 1
                # TODO: update file status.
                c_time = datetime.fromtimestamp(os.path.getctime(file_name))
                record_upload_file_status(LiveFile.STATUS_UPLOADED, base_name,vmix_name, c_time)
        ftp.quit()

    except Exception as e:
        logger.error(e)
    finally:
        # update job status.
        if upload_counter == files_amount:
            record_upload_job_status(upload_job_id, UploadJob.STATUS_UPLOAD_SUCCESS)
        else:
            record_upload_job_status(upload_job_id, UploadJob.STATUS_UPLOAD_FAILED)


def record_upload_job_start_in_db(files_to_upload, vmix_name):
    # Create upload_job record
    record_upload_job(vmix_name, len(files_to_upload))
    upload_job_id = get_latest_upload_job(vmix_name)
    logger.info("Created upload job. id=%d" % upload_job_id)

    # Insert file info into video_file table
    for file_name in files_to_upload:
        file_base_name = path.basename(file_name)
        file_info = LiveFile(upload_job_id, vmix_name, file_base_name)
        file_info.size = path.getsize(file_name)

        file_info.c_time = datetime.fromtimestamp(os.path.getctime(file_name))
        file_info.m_time = datetime.fromtimestamp(path.getmtime(file_name))
        path.getctime(file_name)
        record_file_info(file_info)

    return upload_job_id


def reschedule_upload_video_files(hour, minute):
    scheduler.reschedule_job(job_id=VIDEO_UPLOAD_JOB_ID, trigger='cron', day_of_week='*', hour=hour, minute=minute)


def schedule_upload_video_files():
    hour = int(SETTINGS['upload_hour'])
    minute = int(SETTINGS['upload_minute'])
    # scheduler.add_job(upload_files, 'cron', id=VIDEO_UPLOAD_JOB_ID, day_of_week='*', hour=hour, minute=minute)
    scheduler.add_job(upload_files, 'interval', seconds=1)
    scheduler.start()


class FtpUploadTracker:
    file_name = ''
    total_size = 0
    block_size = 0
    uploaded_size = 0
    percent = 0

    def __init__(self, file_name, total_size, block_size):
        self.file_name = file_name
        self.total_size = total_size
        self.block_size = block_size

    def handle(self, block):
        self.uploaded_size += len(block)
        self.percent = round((self.uploaded_size / self.total_size) * 100)
        logger.info("uploaded percent: %d" % self.percent)
        if self.uploaded_size >= self.total_size:
            logger.info("%s %d is uploaded!" % (self.file_name, self.total_size))


if __name__ == '__main__':
    SETTINGS = get_local_setting()
    logger.info('Current local setting: \n%s' % SETTINGS)

    schedule_upload_video_files()

    app.run(host='0.0.0.0', port=5000, debug=False)
