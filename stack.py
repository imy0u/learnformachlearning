#栈，堆栈的操作，仅限在标为进行插入和删除的线性表
class Stack(object):
    #初始化
    def __init__(self) -> None:
        self.items=[]
    
    def clearstack(self):
        self.items.clear()

    #判断栈是否为空,返回布尔值
    def is_empty(self):
        return self.items==[]
    
    #返回栈顶元素
    def gettop(self):
        if self.is_empty():
            print('栈为空，非法操作！')
            return False
        else:
            return self.items[-1]#表示数组最后一个
        # Python中数组的用法灵活多样，常用的记下来了，但是遇到-1就容易混淆。在这里记录一下。一个数组a=[0,1,2,3,4]，a[-1]表示数组中最后一位，a[:-1]表示从第0位开始直到最后一位，a[::-1]表示倒序，从最后一位到第0位。
    #返回栈的大小
    def size(self):
        return(len(self.items))
    #进栈
    def push(self,item):
        self.items.append(item)#在末尾插入
    #出栈
    def pop(self):
        if self.is_empty():
            return False
        else:
            return self.items.pop()#用于移除列表中的一个元素（默认是最后一个元素）
    