#图，图是顶点的集合，这些顶点通过一系列边结对链接。
#图的遍历和图的最短路径算法
#图的遍历算法有两种算法，深度优先和广度优先
#深度优先算法，类似于二叉树中的先序遍历，
#广度优先算法，类似于二叉树中的层序遍历
class Graph(object):
    #使用邻接表存储图
    def __init__(self,kind):
        #kind代表图的类型，无向图，有向图，无向网，有向网
        #kind:Undigraph,Digraph,Undinetwork,Dinetwork
        self.kind=kind
        self.vertices=[]#邻接表
        self.vexnm=0#顶点数
        self.arcnum=0#当前边数
        #具体的结构是什么样的？
    def CreateGraph(self,vertex_list,edge_list):
        #创建图
        """
        创建图
        param vertex_list:顶点列表
        param edge_list:边列表
        """
        self.vexnum=len(vertex_list)#顶点数
        self.aecnum=len(edge_list)#边数
