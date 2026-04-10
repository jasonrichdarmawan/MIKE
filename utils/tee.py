# %%

import logging
import sys
import copy

# %%

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':  # avoid printing bare newlines, if you like
            self.logger.log(self.level, message)

    def flush(self):
        # doesn't actually do anything, but might be expected of a file-like
        # object - so optional depending on your situation
        pass

    def close(self):
        # doesn't actually do anything, but might be expected of a file-like
        # object - so optional depending on your situation. You might want
        # to set a flag so that later calls to write raise an exception
        pass
    
class BufferingLoggerWriter(LoggerWriter):
    def __init__(self, logger, level):
        super().__init__(logger, level)
        self.buffer = ''

    def write(self, message):
        if '\n' not in message:
            self.buffer += message
        else:
            parts = message.split('\n')
            if self.buffer:
                s = self.buffer + parts.pop(0)
                self.logger.log(self.level, s)
            self.buffer = parts.pop()
            for part in parts:
                self.logger.log(self.level, part)

class MultilineMixin:
    def emit(self, record):
        s = record.getMessage()
        if '\n' not in s:
            super().emit(record)
        else:
            lines = s.splitlines()
            rec = copy.copy(record)
            rec.args = None
            for line in lines:
                rec.msg = line
                super().emit(rec)
                
class StreamHandler(MultilineMixin, logging.StreamHandler):
    pass

class FileHandler(MultilineMixin, logging.FileHandler):
    pass
    
# %%

if __name__ == "__main__":
    fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)-9s %(message)s')
    
    sh = StreamHandler()
    sh.setFormatter(fmt)
    
    fh = FileHandler("output.log", mode='w')
    fh.setFormatter(fmt)
    
    logger = logging.getLogger("demo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    # Redirect stdout to the logger
    sys.stdout = BufferingLoggerWriter(logger, logging.INFO)
    sys.stderr = BufferingLoggerWriter(logger, logging.WARNING)
    
# %%
    
if __name__ == "__main__":
    # Example usage
    print("from print statement")
    logger.debug('Single line')
    logger.debug('Multiple lines:\nnext line')
    logger.debug('Another single line')
    logger.debug('Multiple lines:\n%s', 'next line\nnext line')
    # Note: if you run this code in Python Interactive Window, 
    # this will appear in the interactive window. However, it will not be stored in the output.log file.
    # Note: if you run this code in a regular Python environment, this will be stored in the output.log file.
    raise ValueError("This is an error message\nwith multiple lines\nand should be logged properly.")

# %%
