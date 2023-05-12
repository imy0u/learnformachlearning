class Animal(object):
    def run(self):
        print('animal is running.....')

class Dog(Animal):
    pass

class Cat(Animal):
    pass
dog=Dog()
dog.run()
cat=Cat()
cat.run()