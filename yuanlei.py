#使用元类
class Hello(object):
    def hello(self,name='world'):
        print('Hello,%s' %name)

h=Hello()
h.hello()
print(type(h))