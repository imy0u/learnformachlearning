import os
print('process (%s) start ......' % os.getpid())
pid=os.F_OK
if pid ==0:
    print('i am child process (%s) and my parent is %s \n.'%((os.getpid),os.getppid()))
else:
    print('i (%s) just created a child process (%s). \n'%(os.getpid(),pid))