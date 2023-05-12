class Animal(object):
    pass

class Runable(object):
    def run(self):
        print('running---')

class Flyable(object):
    def fly(self):
        print('flying---')
# 大类:
class Mammal(Animal):
    pass

class Bird(Animal):
    pass

# 各种动物:
class Dog(Mammal,Runable):#多重继承
    pass

class Bat(Mammal,Flyable):
    pass

class Parrot(Bird):
    pass

class Ostrich(Bird):
    pass

print(Bird('xiaoxiaoniao'))