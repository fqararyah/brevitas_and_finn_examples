
class Settings:
    delimiter = '::'
    # Instance attribute
    def __init__(self, file_name):
        self.file_name = file_name
        with open(file_name, 'r') as f:
            for line in f:
                line = line.replace(' ','').replace('\n', '')
                splits = line.split(self.delimiter)
                if splits[0] == 'out_dir':
                    self.out_dir = splits[1]
                else:
                    setattr(self, splits[0], splits[1])