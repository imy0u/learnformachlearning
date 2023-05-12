# def foo(s):
#     n=int(s)
#     assert n!=0,'n is zero!'
#     return 10/n
# def main():
#     print(foo('0'))
    
# import logging
# logging.basicConfig(level=logging.info)

# s='0'
# n=int(s)
# logging.info('n=%d' %n)
# print(10/n)

#err.py
import pdb

s='0'
n=int(s)
pdb.set_trace()
print(10/n)
