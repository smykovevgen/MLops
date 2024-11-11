class NameKeyError(KeyError):
    def __init__(self, text):
        self.txt = text


class AlreadyExistsError(Exception):
    def __init__(self, text):
        self.txt = text


class ParamsTypeError(TypeError):
    def __init__(self, text):
        self.txt = text


class InvalidData(Exception):
    def __init__(self, text):
        self.txt = text


class ConnectionError(Exception):
    def __init__(self, text):
        self.txt = text
