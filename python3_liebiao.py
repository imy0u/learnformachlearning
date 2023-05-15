#python3 数据结构
#时间：2023、5、15
#地点：五楼
#内容：列表操作的基本函数
#在python中 列表是可变的，字符串和元组不能变动
a = [66.25, 333, 333, 1, 1234.5]
print(a.count(333))
#count意思是返回在列表中出现的次数
a.insert(2,-1)
#在2号位置插入-1
a.append(999)
#在列表末尾添加999，相当于insert(len(a),999)
a.index(333)
#返回列表中第一个等于333的索引，通俗的说就是333在哪里
a.remove(333)
#删除第一个等于333的元素
a.reverse()
#reverse中文为倒置的含义
a.sort()
#对列表中的元素进行排序，从小到大，
print(a)
# 将列表当作堆栈使用，也就是只对列表的头和尾进行操作
#在末尾添加append
#pop从列表的指定位置移除元素，并将其返回。如果没有后指定索引，则返回最后一个元素
stack = [1,2,3,4,5,6]
stack.append(7)
#在栈顶插入元素
print(stack)
stack.pop()
#在栈顶删除元素
print(stack)
#将列表当作队列使用
#在队列中，队尾添加元素append
#对头删除元素remove
#在指导手册中，他用了以下的方法
from collections import deque
queue=deque(['小胡','小黄','雷雷'])
print(queue)
queue.append('豆豆')
print(queue)
queue.popleft()
print(queue)
# 列表推导式
vec=[2,4,6,8]
vec1=[1,3,5,7]
print([3*x for x in vec])
print([[x,x**2] for x in vec])
freshfruit=['   banana   ','   loganberry   ','   passion fruit   ']
print([weapon.strip() for weapon in freshfruit])
# strip方法用于移除字符串头尾指定的字符或字符序列
print([3*x for x in vec if x>3])
print([x*y for x in vec for y in vec1])
print([str(round((355/113),i)) for i in range(1,6)])
# round 表达的是小数点后的位数
# 嵌套列表，矩阵，矩阵的转置
matrix=[
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
]
print([[row[i] for row in matrix] for i in range(4)])
# 元组由若干逗号分隔的值组成，
t=122,132,133,134,'lkjhg'
print(t[0])
print(t)
u=t,(1,2,3,4,5)
print(u)
#元组在输出时有括号的，以便于正确表达嵌套结构，在输入时可能有或没有括

#集合
#集合是一个无序不重复元素的集。基本功能包括关系测试和消除重复元素
fruit={'apple','orange','pear','banana','apple'}
print(fruit)
#输出自动剔除重复
print('apple'in fruit)
#python set函数
#创建一个无序不重复元素集合，可以计算交并补
a=set('jbhqwernjkijhbnjijhb')
b=set('kjhbnjqiujhbnijhn')
print(a)
print(a-b)
print(a&b)#交集
print(a^b)#a或b中有的字母，不同时在 ab交集的不记
print(a|b)#在a或b中的，包括同时在的 并集
#字典
#无序的键值对集合，关键字之间必须是互不相同的
tel={'jack':133333333,'sape':413939393999}
tel['guido']=41272727777
print(tel)
print(list(tel.keys()))
#字典的遍历，关键字和对应的值可以使用items方法同时解读出来
knights={'gallahad':'the pure','robin':'the brave'}
for k,v in knights.items():
    print(k,v)
#enumerate()函数 获得的是索引位置和对应值
for i ,v in enumerate(['tic','tac','toe']):
    print(i,v)
questions=['name','quest','favorite color']
answers=['lancelot','the holy grail','blue']
for q,a in zip(questions,answers):#同时遍历两个或更多的序列，可以使用zip（）
    print('what is your {0} it is {1}.'.format(q,a))