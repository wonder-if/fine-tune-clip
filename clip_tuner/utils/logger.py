import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime


class Logger:
    def __init__(
        self, log_dir="logs", log_file="app.log", file_mode="a", log_level=logging.DEBUG
    ):
        """
        初始化日志系统
        :param log_dir: 日志文件保存的目录
        :param file_mode: 日志文件的打开模式，默认为追加模式
        :param log_level: 日志级别，默认为DEBUG
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志文件名
        self.log_file = os.path.join(log_dir, log_file)

        # 设置日志格式
        log_format = "%(asctime)s [%(levelname)s] [%(filename)s] [%(funcName)s] [line %(lineno)d] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # 配置日志记录器
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # 添加文件处理器（按天分割）
        file_handler = TimedRotatingFileHandler(
            self.log_file, when="midnight", interval=1, backupCount=7
        )
        file_handler.mode = file_mode
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        self.logger.addHandler(file_handler)

        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        self.logger.addHandler(console_handler)

    @property
    def log_path(self):
        return os.path.abspath(os.path.join(os.getcwd(), self.log_file))

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


# 示例用法
if __name__ == "__main__":
    # 初始化日志系统
    logger = Logger()

    # 记录不同级别的日志
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
