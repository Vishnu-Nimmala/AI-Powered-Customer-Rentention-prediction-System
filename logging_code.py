import logging
import sys


def setup_logging(script_name):
    try:
        logger = logging.getLogger(script_name)

        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(f'C:\\Users\\vishnu\\Desktop\\Churn_prediction\\logs\\{script_name}.log',mode='w')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        return logger


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')