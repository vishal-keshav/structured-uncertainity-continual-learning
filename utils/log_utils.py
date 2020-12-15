import os
import uuid
from comet_ml import Experiment

class logger:
    """logger interface class
    All loggers implements the method of this class
    """
    def __init__(self, **kwargs):
        self.exp_name = kwargs.get('exp_name', str(uuid.uuid1()))
    
    def set_exp_name(self, exp_name):
        self.exp_name = exp_name
    
    def get_exp_name(self):
        return self.exp_name

    def log(self, tag, value, **kwargs):
        """Log value with a given tag

        Args:
            tag (str): Tag associated to the value.
            value (any): Value to be logged.
        """
        raise NotImplementedError
    
    def info(self, msg, **kwargs):
        """Insert information which are not relevant to the experiments.
        A default implementation prints the information on screen, but specific
        loggers can override this functionality

        Args:
            msg (str): A string
        """
        print(msg)

class no_logger(logger):
    """This is no logger version that mocks the logger but does nothing.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def log(self, tag, value, **kwargs):
        pass

    def info(self, msg, **kwargs):
        pass

class print_logger(logger):
    """A simple print based logger.
    Instead of using prints, use log.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log(self, tag, value, *kwargs):
        print(tag + ": " + str(value))

class file_logger(logger):
    """A simple logger that writes everything in a file.
    The file is located in the logs folder, with file name same as exp_name.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dir_path = os.path.join('logs', self.exp_name)
        os.makedirs(dir_path)
        self.fd = open(os.path.join(dir_path, 'logs.txt'), 'a+')
    
    def log(self, tag, value, *kwargs):
        self.fd.write(tag + ": " + str(value) + '\n')
    
    def info(self, msg, **kwargs):
        self.fd.write(msg + '\n')


class comet_logger(logger):
    """Comel.ml logger.
    Use this logger if experiments are extensively complicated.
    """
    def __init__(self, **kwargs):
        self.exp = Experiment(**kwargs)
        super().__init__(**kwargs)
    
    def log(self, tag, value, **kwargs):
        self.exp.log_metric(tag, value, **kwargs)

class chained_loggers(logger):
    """Chaining loggers to combine multiple logging facilities
    """
    def __init__(self, **kwargs):
        self.loggers = []
        super().__init__(**kwargs)
    
    def log(self, tag, value, **kwargs):
        for l in self.loggers:
            l.log(tag, value, **kwargs)
    
    def add_logger(self, l):
        self.loggers.append(l)

def get_logger(args):
    if args.logger is None: return no_logger()
    if args.logger == 'print':
        return print_logger()
    if args.logger == 'file':
        return file_logger()
    if args.logger == 'comet':
        return comet_logger(api_key=args.api_key)
    if args.logger == 'print_file':
        l = chained_loggers()
        l.add_logger(print_logger())
        l.add_logger(file_logger())
        return l