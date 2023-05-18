#二叉树
#二叉树是一种树形结构，每个节点最多只有两个子树主要内容包括
#遍历（前序遍历、中序遍历、后序遍历、层次遍历） 插入和删除
class Node(object):
    def __init__(self,elem=-1,lchild=None,rchild=None) -> None:
        self.elem=elem
        self.lchild=lchild
        self.rchild=rchild
        #python中，可以用这种方式设置初值

class Tress(object):
    def __init__(self) -> None:
        self.root=Node()
        self.myQueue=[]#这里定义是什么结构
    def add(self,elem):
        #插入节点
        node=Node(elem)
        if self.root.elem==-1:#代表树空
            self.root=node
            self.myQueue.append(self.root)
        else:
            treeNode=self.myQueue[0]
            if treeNode.lchild==None:
                treeNode.lchild=node
            else:
                treeNode.rchild=node
                self.myQueue.append(treeNode.rchild)
                self.myQueue.pop(0)#这里没搞明白
    
    def front_digui(self,root):
        #递归先序遍历
        if root==None:
            print('树为空，无法遍历')
        print(root.elem)
        self.front_digui(root.lchild)
        self.front_digui(root.rchild)
    
    def middle_digui(self,root):
        #递归中序遍历
        if root==None:
            print('树为空，无法进行遍历！')
        self.middle_digui(self.lchild)
        print(root.elem)
        self.middle_digui(self.rchild)
    def later_digui(self,root):
        #递归后序遍历
        if root==None:
            print('树为空，无法进行遍历！')
        self.later_digui(self.lchild)
        self.later_digui(self.rchild)
        print(root.elem)

    def front_stack(self,root):
        #利用堆栈实现树的先序遍历
        #首先考虑先序实现的步骤：
        #有左子树就访问左子树，没有左子树就进栈根节点
        #想不明白的时候，就去按照逻辑去一步步写下来
        #按照根左右的顺序进栈，如果有左右节点，根节点出栈后，左右节点分别进栈
        #打印自身，入栈访问左孩子，出栈访问右孩子
        if root==None:
            print('树为空，无法进行遍历！')
        myStack=[]
        node=root
        while node or myStack:
            while node:
                print(node.elem)
                myStack.append(node)
                node=node.lchild
            node=myStack.pop
            node=node.rchild
    
    def middle_stack(self,root):
        #思维重要的是出入栈的顺序，其次才是打印
        if root==None:
            print('树为空，无法进行遍历！')
        myStack=[]
        node=root
        while node or myStack:
            while node:
                myStack.append(node)
                node=node.lchild
            node=myStack.pop
            print(node.elem)
            myStack.append(node.rchild)

    def later_stack(self,root):
        #后续遍历，有点麻烦
        if root==None:
            print('树为空，无法进行遍历！')
        myStack=[]
        node=root
        while node or myStack:
            while node:
                myStack.append(node)
                node=node.lchild
                
    
    def level_queue(self,root):
        #层次遍历，用队列
        if root==None:
            print('树为空，无法进行遍历！')
        myQueue=[]#队列
        node=root
        myQueue.append(node)
        while node or myQueue:
            #myQueue.append(node)
            node=myQueue.pop(0)
            print(node.elem)
            if node.lchild !=None:
                myQueue.append(node.lchild)
            if node.rchild !=None:
                myQueue.append(node.rchild)
