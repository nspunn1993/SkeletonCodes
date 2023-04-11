import datetime
import os

class GenerateLogs:
    def __init__(self, path, mode='w', flag = True):
        self.path = path
        self.mode = mode
        file_path = path[:path.rfind('/')]
        self.checkncreate_dir(file_path)
        file = open(self.path, self.mode)
        if flag:
            self.print_log('Initiating the logs --------------------')
        file.close()
    
    def checkncreate_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def print_log(self, in_str):
        file = open(self.path, self.mode)
        timestamp = datetime.datetime.now()
        
        print(timestamp, end = ' - ', file=file)
        print(str(in_str), file=file)

        file.close()

        print(timestamp, end = ' - ')
        print(str(in_str))