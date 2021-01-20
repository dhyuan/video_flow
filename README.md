# video_flow



dev environment

// 运行mysql server docker
docker run --name MyDB -v /tmp/video_flow/mysqlcnf:/etc/mysql/conf.d   -v /tmp/video_flow/vmix_service/db:/var/lib/mysql -p3306:3306 -p33060:33060 -e MYSQL_ROOT_PASSWORD=password -it --rm mysql:8.0.21

// 运行mysql client 连接到mysql docker ‘c2a3b1413ae4’
docker run -it --network container:c2a3b1413ae4  --rm mysql mysql -hc2a3b1413ae4 -uroot -p


CREATE SCHEMA `vmix_video` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin ;

CREATE TABLE `vmix_setting` (
  `vmix_name` varchar(100) COLLATE utf8_bin NOT NULL,
  `vmix_host` varchar(100) COLLATE utf8_bin DEFAULT NULL,
  `ftp_user` varchar(50) COLLATE utf8_bin DEFAULT NULL,
  `ftp_pwd` varchar(50) COLLATE utf8_bin DEFAULT NULL,
  `ftp_server` varchar(100) COLLATE utf8_bin DEFAULT NULL,
  `uplode_hour` int DEFAULT NULL,
  `upload_minute` int DEFAULT NULL,
  PRIMARY KEY (`vmix_name`),
  UNIQUE KEY `id_UNIQUE` (`vmix_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
  
INSERT INTO `vmix_video`.`vmix_setting` (`vmix_name`, `ftp_user`, `ftp_pwd`, `ftp_server`, `uplode_hour`, `upload_minute`, `vmix_host`) VALUES ('vmix-1', 'vmix-1', 'password', 'localhost', '23', '20', 'localhost');


CREATE TABLE `upload_job` (
  `id` int NOT NULL AUTO_INCREMENT,
  `files_amount` int DEFAULT NULL,
  `from_client` varchar(200) COLLATE utf8_bin DEFAULT NULL,
  `start_time` datetime DEFAULT CURRENT_TIMESTAMP,
  `end_time` datetime DEFAULT NULL,
  `status` int DEFAULT '0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin


CREATE TABLE `live_file` (
  `id` int NOT NULL AUTO_INCREMENT,
  `file_name` varchar(45) COLLATE utf8_bin NOT NULL,
  `size` int DEFAULT '0',
  `status` int DEFAULT '0',
  `from_client` varchar(200) COLLATE utf8_bin DEFAULT NULL,
  `c_time` datetime DEFAULT NULL,
  `m_time` datetime DEFAULT NULL,
  `tried_times` int DEFAULT '0',
  `convert_start_time` datetime DEFAULT NULL,
  `convert_end_time` datetime DEFAULT NULL,
  PRIMARY KEY (`id`,`file_name`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin





# video_flow
