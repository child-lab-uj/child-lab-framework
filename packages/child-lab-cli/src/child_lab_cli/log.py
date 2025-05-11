import logging
import logging.config


def setup_logging() -> None:
    for package in BLACKLIST:
        logging.getLogger(package).setLevel(logging.WARNING)

    logging.config.dictConfig(LOGGER_CONFIG)


BLACKLIST = ['websockets']

LOGGER_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'colored': {
            '()': 'colorlog.ColoredFormatter',
            'fmt': '%(log_color)s[%(levelname)s]%(reset)s %(pathname)s %(message)s',
            'log_colors': {
                'DEBUG': 'blue',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
        },
        'simple': {'format': '[%(levelname)s] %(message)s'},
    },
    'handlers': {
        'stderr': {
            'class': 'logging.StreamHandler',
            'formatter': 'colored',
            'stream': 'ext://sys.stderr',
        }
    },
    'loggers': {
        'root': {
            'level': 'NOTSET',  # capture everything
            'handlers': ['stderr'],
        }
    },
}
