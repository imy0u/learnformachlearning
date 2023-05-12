from enum import Enum,unique

Month = Enum('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
#枚举 枚举名 枚举元素
for name, member in Month.__members__.items():
    print(name, '=>', member, ',', member.value)

@unique#帮助检查保证没有重复值
class Weekday(Enum):
    Sun=0
    Mon=1    
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6

day1=Weekday.Mon
print(day1)
for name,member in Weekday.__members__.items():
    print(name,'=>',member,'-->',member.value)

class Gender(Enum):
    Male=0
    Female=1

class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = Gender(gender)
        
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')