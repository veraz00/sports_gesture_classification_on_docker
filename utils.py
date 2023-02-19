import logging 

def set_logger(file_name = 'record.log'):
    if file_name.endswith('.log'):
        file_name = file_name[:-4]

    logging.basicConfig(filename=f'{file_name}' + '.log', \
                        filemode='w',\
                        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', \
                        datefmt='%Y-%b-%d:%H:%M:%S', \
                        level=logging.DEBUG)  # format: {name}s-- would be the logger name

    logger = logging.getLogger(file_name)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger 


