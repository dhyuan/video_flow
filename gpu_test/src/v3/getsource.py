#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
from datetime import datetime,timedelta

from MyDatabase import *

db = MyDatabase("zghtai")

class VideoSource(object):
    id = 0
    name = ""
    category = ""
    duration = 0.0
    filepath = ""
    filesize = 0
    starttime = 0
    finishtime = 0
    log = ""
    result = ""
    listsize = 0
    resultpath = ""
    create_time = ""
    update_time = ""


STATUS_CREATE = 0  # '待处理'
STATUS_DOING = 1  # '正在处理'
STATUS_SUCCESS = 2  # '处理成功'
STATUS_FAILED = 99  # '处理失败'

class MySource(object):
    _db = None
    _lastid = 0

    def __init__(self):
        super(MySource, self).__init__()
        _db = db

    def getOne(self):
        sqlsel = 'select id, name, filepath from a_source_1 where status = {} order by id asc limit 1'.format(STATUS_CREATE)
        print(sqlsel)
        res = db.fetchone(sqlsel)
        if not res:
            expiretime = (datetime.now()- timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
            sqlsel = 'select id, name, filepath from a_source_1 where status = {} and starttime < "{}" order by id asc limit 1'\
                .format(STATUS_DOING, expiretime)
            print(sqlsel)
            res = db.fetchone(sqlsel)
        if res:
            update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sqlupdate = 'update a_source_1 set status={} , starttime = "{}", update_time="{}" where id = {}' \
                .format(STATUS_DOING, update_time,update_time, res['id'])
            #print(sqlupdate)
            logtime(sqlupdate)
            if db.update(sqlupdate):
                logtime("Get One Success with id = {} name = {}".format(res['id'], res['name']))
            else:
                logtime("Get One Failed")
                res = None
        return res

    def text_read(self,filename):
        # Try to read a txt file and return a matrix.Return [] if there was a mistake.
        try:
            file = open(filename, 'r')
        except IOError:
            error = []
            return error
        content = file.readlines()
        file.close()
        return content

    def finishOne(self,id, log="", resultfile="", flag=True):
        sqlsel = 'select id, name, filepath from a_source_1 where status = {} order by id asc limit 1'.format(
            STATUS_DOING)
        print(sqlsel)
        res = db.fetchone(sqlsel)
        if not res:
            print("Not found  doing status source with id={}".format(id) )
            logtime("Not found  doing status source with id={}".format(id))
            return False
        update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if flag:
            status = STATUS_SUCCESS
        else:
            status = STATUS_FAILED
        result = ""
        listsize = 0
        if os.path.exists(resultfile):
            lines = self.text_read(resultfile)
            listsize = len(lines)
            result = result.join(lines)
        if listsize <= 0:
            logtime("file %s result is empty ..." % resultfile)
            status = STATUS_FAILED
        sqlupdate = 'update a_source_1 set status={} , log="{}", resultpath="{}", result="{}", ' \
                    ' listsize = {}, finishtime = "{}", update_time="{}" where id = {}' \
                    .format(status, log, resultfile, result, listsize, update_time, update_time, id)
        #print(sqlupdate)
        #logtime(sqlupdate)
        if db.update(sqlupdate):
            logtime("Update successful!")
            return True
        else:
            logtime("Update failed!")
            return False

    def getOneResult(self):
        sqlsel = 'select id, name, result from a_source_1 where id > {} and status = {} order by id asc limit 1' \
            .format(self._lastid, STATUS_SUCCESS)
        logtime(sqlsel)
        res = db.fetchone(sqlsel)
        if res:
            self._lastid = res['id']
            logtime("Get result of [%d] %s successful!" % (res['id'],res['name']))
        else:
            logtime("Get none result.")
        return res

    def getAllOkIds(self, startid=0, endid=0):
        if endid > startid:
            sqlsel = 'select id, name, finishtime from a_source_1 where (id between {} and {} ) and status = {} order by id asc ' \
                .format(startid,endid, STATUS_SUCCESS)
        else:
            sqlsel = 'select id, name, finishtime from a_source_1 where id > {} and status = {} order by id asc ' \
                .format(startid, STATUS_SUCCESS)
        try:
            res = db.fetchall(sqlsel)
            if res:
                return res
            else:
                return None
        except Exception as ex:
                logtime(ex)
                return None

    def getOneOkById(self, id):
        sqlsel = 'select id, name, finishtime, result from a_source_1 where id = {} and status = {} order by id asc limit 1' \
            .format(id, STATUS_SUCCESS)
        try:
            res = db.fetchone(sqlsel)
            if res:
                return res
            else:
                return None
        except Exception as ex:
                logtime(ex)
                return None