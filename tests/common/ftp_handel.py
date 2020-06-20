# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" ftp relation operations

"""
import socket
import os
from ftplib import FTP


class ftpHandle():
    def __init__(self, logger, timeout=180, buf_size=8192):
        self.ftp_instance = None
        self._log = logger
        self.host = os.environ.get("FTP_HOST")
        self.timeout = timeout
        self.buf_size = buf_size

    def ftp_login(self):
        self.ftp_instance = FTP()

        if not self.host:
            self._log.error("ftp_login :: ftp host is not set environment variable value")
            return False

        try:
            self.ftp_instance.connect(host=self.host, timeout=self.timeout)
            self._log.info("ftp_login:: ftp connect successful for host : {0}".format(self.host))
        except (socket.error, socket.gaierror)as e:
            self._log.error("ftp_login:: ftp connect faile for host : {0}, msg:{1}".format(self.host, e))
            return False

        try:
            self.ftp_instance.login()
        except Exception as e:
            self._log.error("ftp_login:: ftp connect faile for host : {0}, msg:{1}".format(self.host, e))
            return False
        return True

    def ftp_mkdir(self, remote_path, dir_name):
        # 判断目录是否存在，存在则不需要重复创建
        try:
            abs_dir = os.path.join(remote_path, dir_name)
            self.ftp_instance.cwd(abs_dir)
            self._log.info("abs_dir : {0} is exist, please not retry".format(abs_dir))
            return True
        except:
            try:
                self.ftp_instance.cwd(remote_path)
                self.ftp_instance.mkd(dir_name)
            except Exception as e:
                self._log.error(
                    "ftp_mkdir:: ftp mkdir faile for host : {0}, new_dir_name: {1},msg:{2}".format(self.host, dir_name,
                                                                                                   e))
                return False
        return True

    def ftp_upload_file(self, remote_path, remote_file_name, local_file_path):

        try:
            self.ftp_instance.cwd(remote_path)
            with open(local_file_path, 'rb') as fp:
                self.ftp_instance.storbinary("STOR " + remote_file_name, fp, self.buf_size)
        except Exception as e:
            self._log.error(
                "ftp_upload_file:: ftp upload faile for host : {0}, remote_epath: {1},local_file_path:{2}, msg:{3}".format(
                    self.host, remote_path, local_file_path, e))
            return False
        return True

    def ftp_download_file(self, remote_path, remote_file_name, local_file_path):
        try:
            self.ftp_instance.cwd(remote_path)
            with open(local_file_path, 'wb') as fp:
                self.ftp_instance.retrbinary("RETR " + remote_file_name, fp.write, self.buf_size)
        except Exception as e:
            self._log.error(
                "ftp_download_file:: ftp download faile for host : {0}, remote_epath: {1},local_file_path:{2}, msg:{3}".format(
                    self.host, remote_file_name, local_file_path, e))
            return False
        return True

    def ftp_close(self):
        self.ftp_instance.quit()
