import logging
import sys


# logging.getLogger().addHandler(logging.StreamHandler())
#
# consoleHandler = logging.StreamHandler(sys.stdout)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(processName)s %(filename)s:%(lineno)s -- %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logging.info('Useful message')
logging.debug('debug message')

logging.error('Something bad happened')