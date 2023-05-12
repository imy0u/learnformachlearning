# #danyuanceshi 
# f=open('/Users/cas/test.txt','r')
# print(f.read())
# f.close

# try:
#     f=open('/Users/cas/test.txt','r',errors='ignore')
#     print(f.read())
# finally:
#     if f:
#         f.close

# with open('/Users/cas/test.txt','w',errors='ignore') as f:
#     f.write('我是你!!!')
#string io
#数据读写的不一定是文件，也可以在内存中读写
# from io import StringIO
# f=StringIO()
# f.write('hello')
# f.write('    ')
# f.write('world!!!')
# print(f.getvalue())
from io import StringIO
f=StringIO('hello!!\nhi\ngoodbye!')
while True:
    s=f.readline()
    if s=='':
        break
    print(s.strip())