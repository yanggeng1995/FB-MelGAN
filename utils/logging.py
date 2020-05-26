import logging

def GetLogging(logfile):
   logger = logging.getLogger(__name__)
   logger.setLevel(level=logging.INFO)

   handler = logging.FileHandler(logfile)
   handler.setLevel(logging.INFO)
   formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
   handler.setFormatter(formatter)
   logger.addHandler(handler)

   console = logging.StreamHandler()
   console.setLevel(logging.INFO)
   console.setFormatter(formatter)
   logger.addHandler(console)

   return logger
