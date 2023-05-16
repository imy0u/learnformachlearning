#线性表的链式表示
class Node (object):
    #定义一个节点
    def __init__(self,elem):
        self.elem=elem
        self.next=None#初始化下一节点为空

#下面创建单链表
class SingleLinkList(object):
    #单链表
    def __init__(self,node=None):
        self._head=node

    def init_list(self,data):
        self._head=Node(data[0])
        p=self._head#指针指向头节点
        for i in data[1:]:
            p.next=Node(i)#确定指针指向下一个结点
            p=p.next#指针滑动向下一个位置
    
    def is_empty(self):
        # '''判断链表是否为空'''
        return self._head==None
    def length(self):
        # 链表长度
        cur=self._head
        count=0
        while cur !=None:
            count+=1
            cur=cur.next
        return count
    def travel(self):
        # 遍历整个链表
        cur=self._head
        while cur !=None:
            print(cur.elem,end='=>')
            cur=cur.next
        print('null\n')
    def addhead(self,item):
        #在链表头添加元素
        node=Node(item)
        node.next=self._head
        self._head=node
    
    def append(self,item):
        #在链表尾部添加元素
        node =Node(item)
        #在链表为空的时候没有next 判断是否为空
        if self.is_empty():
            self._head=node
        else:
            cur=self._head
            while cur.next!=None:
                cur=cur.next
            cur.next=node
    
    def insert(self,pos,item):
        #在指定位置添加元素
        #首先判断插入位置是否合法
        if pos<0 or pos>self.length:
            print('插入位置不合法')
        else:
            per=self._head
            count=0
            while count<pos-1:#这里到底是pos-1还是pos？是pos-1,因为循环结束的时候，是停在pos
                count+=1
                per=per.next
            node=Node(item)
            node.next=per.next
            per.next=node
    
    def remove(self,item):
        #删除节点值等于item的
        #删除节点这里，如果说理论，没啥问题，可是按照别人博客的方法，总觉得怪怪的

        cur=self._head
        pre=None#前驱结点
        while cur!=None:
            if cur.elem==item:
                #考虑这个节点是不是头节点
                #如果不考虑会怎么样？
                if cur==self._head:
                    #是头节点，就把下一个节点赋值给头节点
                    self._head=cur.next#这个时候原本的头节点还没有被释放
                else:
                    #不是头节点
                    #这里需要执行删除操作
                    pre.next=cur.next
                break
            else:
                pre=cur
                cur=cur.next#检测不等于就跳过
    
    def search(self,item):
        #查找某个节点
        cur=self._head
        while cur:
            if cur.elem==item:
                return True
            else:
                cur=cur.next
        return False

if __name__=='__main__':
    L1=SingleLinkList()
    print(L1.is_empty())#判断是否为空
    print(L1.length())
    L1.init_list([1,5,6,65])
    L1.travel()
    L1.append(655)
    L1.addhead(123)
    L1.remove(655)#所有操作里，删除掌握的最差
    L1.travel()