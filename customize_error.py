# 自定义异常 需要继承Exception
class MyException(Exception):

    def __init__(self, *args):
        self.args = args


# 常见做法定义异常基类,然后在派生不同类型的异常
class loginoutError(MyException):
    def __init__(self):
        self.args = ('退出异常',)
        self.message = '退出异常'
        self.code = 200


raise Exception()













