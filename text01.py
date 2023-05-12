std1={'name':'xiaoming','score':100}
std2={'name':'xiaohong','score':99}
def print_score(std):
    print('%s:%s'%(std['name'],std['score']))

class student(object):
    def __init__(self,name,score):
        self.__name=name
        self.__score=score
    
    def print_score(self):
        print('%s:%s'%(self.__name,self.__score))
    
    def set_score(self,score):
        if 0<=score<=100:
            self.__score=score
        else:
            raise ValueError('bad score')
lisa=student('lisa llllssss',99)
dina=student('dina dddnnn',88)
lisa.print_score()
dina.print_score()
lisa.set_score(100)
lisa.print_score()
dina.set_score(12)
dina.print_score()