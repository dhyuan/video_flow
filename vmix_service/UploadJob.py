
class UploadJob:
    STATUS_UPLOAD_START = 0
    STATUS_UPLOAD_SUCCESS = 1
    STATUS_UPLOAD_FAILED = 2
    a = 0

    def __init__(self, start_time, from_client):
        self.start_time = start_time
        self.from_client = from_client


class LiveFile:
    STATUS_WAIT_TO_UPLOAD = 0
    STATUS_UPLOADED = 1
    STATUS_CONVERTED_SUCCESS = 2
    STATUS_CONVERTED_FAILED = 3

    def __init__(self, job_id, from_client, file_name, tags="", size=0, c_time=None, m_time=None):
        self.upload_job_id = job_id
        self.file_name = file_name
        self.tags = tags
        self.size = size
        self.status = 0
        self.from_client = from_client
        self.upload_times = 0
        self.c_time = c_time
        self.m_time = m_time
