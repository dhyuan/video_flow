#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys,os
from datetime import datetime,timedelta

from MyDatabase import *

db = MyDatabase("zghtai")
Tpl_Table = "a_test_"
Tpl_Table_NA = "a_test_na_"

class TestVideo():
    id = 0
    name = ""
    source_id = 0
    source_name = ''
    duration = 0
    filepath = ""
    filesize = 0
    rootpath = ""
    filepath = ""
    filesize = 0
    startpos = 0
    result = ''
    status = 0

STATUS_CREATE = 0  # '待处理'
STATUS_DOING = 1  # '正在处理'
STATUS_SUCCESS = 2  # '处理成功'
STATUS_FAILED = 99  # '处理失败'

def checkTableExists(tablename):
    sql = "show tables"
    db.cursor.execute(sql)
    tables = db.cursor.fetchall()
    # print(tables)
    tables_list = re.findall('(\'.*?\')', str(tables))
    # print(tables_list)
    tables_list = [re.sub("'", '', each) for each in tables_list]
    # print(tables_list)
    if tablename in tables_list:
        print("table %s exists" % tablename)
        return True
    else:
        print("table %s not exists" % tablename)
        return False


class MyTest(object):
    _db = None
    second = 0
    tl_name = ''

    def __init__(self, second=10,flag=True):
        super(MyTest, self).__init__()
        self._db = db
        self.second = second
        if flag:
            self.tl_name = Tpl_Table + str(self.second)
        else:
            self.tl_name = Tpl_Table_NA + str(self.second)
        if not checkTableExists(self.tl_name):
            raise Exception("Table %s not exists" % self.tl_name)



    def getOne(self):
        sqlsel = 'select id, name, filepath from {} where status = {} order by id asc limit 1'.format(self.tl_name, STATUS_CREATE)
        print(sqlsel)
        res = db.fetchone(sqlsel)
        if not res:
            expiretime = (datetime.now()- timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            sqlsel = 'select id, name, filepath from {} where status = {} and starttime < "{}" order by id asc limit 1'\
                .format(self.tl_name, STATUS_DOING, expiretime)
            print(sqlsel)
            res = db.fetchone(sqlsel)
        if res:
            update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sqlupdate = 'update {} set status={} , starttime = "{}", update_time="{}" where id = {}' \
                .format(self.tl_name, STATUS_DOING, update_time,update_time, res['id'])
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

    def finishOneFile(self,id, log="", resultfile="", flag=True):
        result = ""
        listsize = 0
        if os.path.exists(resultfile):
            lines = self.text_read(resultfile)
            listsize = len(lines)
            result = result.join(lines)
        if listsize <= 0 or len(result) == 0:
            logtime("file %s result is empty ..." % resultfile)
            flag = False
        return self.finishOne(id, log,result,flag)


    def finishOne(self,id, log="", result="", flag=True):
        sqlsel = 'select id, name, filepath from {} where status = {} order by id asc limit 1' \
            .format(self.tl_name, STATUS_DOING)
        print(sqlsel)
        res = db.fetchone(sqlsel)
        if not res:
            #print("Not found doing status test file with id={}".format(id) )
            logtime("Not found doing status test file with id={}".format(id))
            return False
        update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if flag:
            status = STATUS_SUCCESS
        else:
            status = STATUS_FAILED
        # result = ""
        # listsize = 0
        # if os.path.exists(resultfile):
        #     lines = self.text_read(resultfile)
        #     listsize = len(lines)
        #     result = result.join(lines)
        if len(result)==0:
            logtime("fresult is empty ...")
            status = STATUS_FAILED
        sqlupdate = 'update {} set status={} , log="{}", result="{}", ' \
                    ' finishtime = "{}", update_time="{}" where id = {}' \
                    .format(self.tl_name, status, log, result, update_time, update_time, id)
        #print(sqlupdate)
        #logtime(sqlupdate)
        if db.update(sqlupdate):
            logtime("Update successful!")
            return True
        else:
            logtime("Update failed!")
            return False


if __name__=='__main__':
    mt = MyTest(6)
    r = mt.getOne()
    print(r)
    mt.finishOne(r['id'],"","test")

    print("finish~~")