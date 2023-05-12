#cuowu
try:
    print('try...')
    r=10/int('a')
    print('result:',r)
except ValueError as e:
    print('valueerror:',e)
except ZeroDivisionError as e:
    print('expect:',e)#没有错误except就不会被执行
finally:
    print('finally!!!')
print('end')
