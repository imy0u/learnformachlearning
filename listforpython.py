# 线性表
# 线性表的顺序表示
import sys

class Lnode(object):
    # 定义节点类
    def __init__(self,last) -> None:
        self.data=None
        self.last=last#定义线性表的长度

#初始化建立空的线性表 
def makeempty(num):
    ptr=Lnode(num)
    return ptr

#查找给定值的位置
def find(x,L):
    i=0
    while(i<=L.last and L.data[i]!=x):
        i+=1
        if(i>L.last):
            return False
        else:
            return i
        
#插入新元素
def insert(x,i,L):
    if i<0 or i>L.last:
        print("位置不合理")
        return False
    else:
        for j in range(L.last,i-1,-1):#从后往前循环
            L.data[j+1]=L.data[j]
        L.data[i]=x
        L.last+=1
    return True


#删除某个位置上的元素
def delete(i,L):
    if i<0 or i>L.last:
        print('删除不合理')
        return False
    else:
        for j in range(i,L.last-1):
            L.data[j]=L.data[j+1]
        L.last-=1
        L.data[L.last+1]=None
    return True

def main():
    while(1):
        print('****************线性表*******************')
        print('一，创建一个新的线性表')
        print('二，插入一个新元素')
        print('三，删除一个元素')
        print('四，查找给定值的元素')
        print('****************************************')
        # print('请输入你的操作（1，2，3，4）')
        i=input('请输入你的操作数（1，2，3，4）:')
        #在python3中input输入是string类型的，一定要注意 不是int类型的
        # i=int(i)
        #python中没有switch函数 使用match函数
        
        if i=='1':
            longth=input('请输入新创建的线性表的长度')
            L=makeempty(longth)
            # return True
        elif i=='2':
            #  insert(x,i,L):插入一个新元素
            x=input("请输入要插入的元素：")
            i=input('请输入要插入的元素位置')
            if insert(x,i,L):
                print('插入成功！')
                # return True
            else:
                print('插入失败！')
                return False
        elif i=='3':
            i=input('请输入要删除元素的位置')
            if delete(i,L):
                print('删除成功！')
                # return True
            else:
                print('删除失败！')
                # return False

        elif i=='4':               
             #find(x,L):给出定值查找元素
            x=input('请输入要查找的元素')
            if find(x,L):
                print(find(x,L))
                print('查找成功！')
                # return True
            else:
                print('查找失败！')
                # return False
        else:
            print('输入不合法，请重新输入')

        #match函数的用法
        # match i:
        #     case 1:
        #         longth=input('请输入新创建的线性表的长度')
        #         L=makeempty(longth)
        #         return True
        #     case 2:
        #         #  insert(x,i,L):插入一个新元素
        #         x=input("请输入要插入的元素：")
        #         i=input('请输入要插入的元素位置')
        #         if insert(x,i,L):
        #             print('插入成功！')
        #             return True
        #         else:
        #             print('插入失败！')
        #             return False
        #     case 3:
        #         #delete(i,L):删除某个位置上的元素
        #         i=input('请输入要删除元素的位置')
        #         if delete(i,L):
        #             print('删除成功！')
        #             return True
        #         else:
        #             print('删除失败！')
        #             return False

        #     case 4:
        #         #find(x,L):给出定值查找元素
        #         x=input('请输入要查找的元素')
        #         if find(x,L):
        #             print(find(x,L))
        #             print('查找成功！')
        #             return True
        #         else:
        #             print('查找失败！')
        #             return False
        #     case _:
        #         print('输入不合法，请重新输入')
        #         return False
            

if __name__=='__main__':
    sys.exit(main())