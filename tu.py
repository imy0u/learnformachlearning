#图，图是顶点的集合，这些顶点通过一系列边结对链接。

# class Graph(object):
#     #使用邻接表存储图
#     def __init__(self,kind):
#         #kind代表图的类型，无向图，有向图，无向网，有向网
#         #kind:Undigraph,Digraph,Undinetwork,Dinetwork
#         self.kind=kind
#         self.vertices=[]#邻接表
#         self.vexnm=0#顶点数
#         self.arcnum=0#当前边数
#         #具体的结构是什么样的？
#     def CreateGraph(self,vertex_list,edge_list):
#         #创建图
#         """
#         创建图
#         param vertex_list:顶点列表
#         param edge_list:边列表
#         """
#         self.vexnum=len(vertex_list)#顶点数
#         self.aecnum=len(edge_list)#边数

# class Vertex:
#     #邻接矩阵构造图的表示
#     def __init__(self,key):
#         self.id=key
#         self.connectedTo={}#这是一个字典
    
#     def addNeighbor(self,nbr,weight=0):#weight没有输入值的时候，默认是hi0
#         self.connectedTo[nbr]=weight
    
#     def __str__(self) -> str:
#         return str(self.id)+'connectedTo:'+str([x.id for x in self.connectedTo])
#     def getConnections(self):
#         return self.connectedTo.keys()  
#     def getId(self):
#         return self.id
#     def getWeight(self,nbr):
#         return self.connectedTo[nbr]
# class Graph():
#     '''邻接矩阵实现图的存储'''
#     def __init__(self) -> None:
#         self.start=[]
#         self.wight=[None]
#         self.count_vertex=0#顶点数
#     def add_vertex(self,key):
#         '''添加顶点'''
#         self.start.append(key)
#         self.count_vertex+=1
#         if self.count_vertex>1:
#             temp_list=[None]*(2*self.count_vertex-1)
#             for i in temp_list:
#                 self.wight.append(i)#这里是什么意思？

#     def add_edge(self,key,nbr,wight=None):
#         '''添加边'''
#         if key not in self.start:
#             self.add_vertex(key)
#         if nbr not in self.start:
#             self.add_vertex(nbr)
#         for i in self.start:
#             if key==i:
#                 index_01=self.start.index(i)+1
#             if nbr==i:
#                 index_02=self.start.index(i)+1#计算在哪里赋值
#         self.wight[index_01*index_02-1]=wight#行乘上列
    
#     def get_vertex_num(self):
#         '''返回所有顶点数量'''
#         return self.count_vertex
    
#     def get_vertex(self,i):
#         '''查找顶点及相应边'''
#         for n in self.start:
#             new_val=[i]
#             if i ==n:
#                 if self.start.index(n)!=0:
#                     for index,val in enumerate(self.wight):
#                         if (index+1)%(self.start.index(n)+1)==0:
#                             if val !=None:
#                                 temp=self.start[(index+1)//(self.start.index(n)+1)-1]#  //表示向负无穷方向取整，例如-5//2=-3
#                                 new_val.append([temp,val])#定义插入的值和位置
#                     return new_val
#                 elif self.start.index(n)==0:
#                     new_list=self.wight[:self.count_vertex]
#                     for i in new_list:
#                         for i in new_list:
#                             if i!=None:
#                                 new_val.append([self.start[new_list.index(i)],i])
#                         return new_val

class Vertex(object):
    '''邻接列表构造矩阵'''
    def __init__(self,key) -> None:
        self.id=key
        self.connect=[]

    def add_neighbor(self,nbr,wight=0): 
        # nbr是顶点对象所连接的另外节点，也就是顶点对象的key值
        # wight表示的权重，也就是两点之间的距离
        self.connect[nbr]=wight

    def get_connects(self):
        return [[i.id,v]for i,v in self.connect.items()]
class Graph(object):
    '''实现图'''
    def __init__(self) -> None:
        self.vertlist={}
        self.count_vertex=0
    def add_vertex(self,key):
        '''在列表中添加节点'''
        self.count_vertex+=1
        self.vertlist[key]=Vertex(key)#还是字典
    def get_vertex(self,i):
        '''查找顶点'''
        return self.vertlist[i].get_connects() if  i in self.vertlist else None
    def add_edge(self,key,nbr,wight=0):
        '''添加边'''
        if key not in self.vertlist:
            self.add_vertex(key)
        if nbr not in self.vertlist:
            self.add_vertex(nbr)
        self.vertlist[key].add_neighbor(self.vertlist[nbr],wight)
    def get_vertex_num(self):
        '''返回所有顶点数量'''
        return self.count_vertex
if __name__ == "__main__":
    graph = Graph()
    graph.add_vertex('a')
    graph.add_vertex('b')
    graph.add_edge('a','b',10)
    graph.add_edge('b','a',10)
    graph.add_edge('a','c',12)
    graph.add_edge('b','c',15)
    print(graph.get_vertex_num())
    print(graph.get_vertex('a'))