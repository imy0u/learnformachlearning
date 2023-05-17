#队列
#队列只允许在表的一端进行插入，在另一端删除，先进先出策略
#链式队列
class Node(object):
    def __init__(self,value) -> None:
        self.data=value
        self.next=None
class Queue(object):
    def __init__(self):
        #定义一个头节点和尾节点
        self.front=Node(None)
        self.rear=self.front
        #初始化，队头和队尾都指向同一节点

    def enQueue(self,element):
        n=Node(element)
        self.rear.next=n
        self.rear=n
        #入队
    
    def deQueue(self):
        #出队
        if self.is_empty():
            print('队空')
            return False
        temp=self.front.next
        self.front.next=self.front.next.next
        if self.rear==temp:
            self.rear=self.front
        del temp

    def getHead(self):
        if self.is_empty():
            return '队空，无值输出'
        return self.front.next.data
    
    def is_empty(self):
        return self.rear==self.front
    
    def printQueue(self):
        #遍历队列
        cur=self.front.next
        tmp=''
        while cur!=None:
            tmp=cur.data
            cur=cur.next
            print(tmp)

    def clear(self):
        #清空队列
        while self.front.next!=None:
            temp=self.front.next
            self.front.next=temp.next#相当于出队操作
        self.rear=self.front

    #计算队长
    def length(self):
        cur=self.front.next
        tmp=''
        i=0
        while cur!=None:
            tmp=cur.data
            cur=cur.next
            i+=1
        return i

if __name__=='__main__':
    queue=Queue()
    queue.enQueue('p')
    queue.enQueue('y')
    queue.enQueue('t')
    queue.enQueue('h')
    queue.enQueue('o')
    queue.enQueue('n')
    queue.printQueue()
    print(queue.length())
    queue.clear()
    print(queue.length())
    queue.printQueue()