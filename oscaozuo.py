import os
# # print(os.environ)
# #操作文件目录
# print(os.path.abspath('.'))
# os.path.join('/Users/cas','testdir')
# #创建一个目录
# #os.mkdir('/Users/cas/testdir')
# os.rmdir('/Users/cas/testdir')

#序列化
import pickle
d=dict(name='bob',age=20,score=88)
print(pickle.dumps(d))
f=open('/Users/cas/test.txt','wb')
d=pickle.load(f)
f.close()
d