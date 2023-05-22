#图的遍历和图的最短路径算法
#图的遍历算法有两种算法，深度优先和广度优先
#深度优先算法，类似于二叉树中的先序遍历，递归和非递归
#广度优先算法，类似于二叉树中的层序遍历
def BFS(graph,s):
    '''
    广度优先,类似于树的层次遍历
    s表示从s这个顶点开始遍历
    '''
    queue=[]#队列
    result=[]
    queue.append(s)
    seen=set()#set函数创建一个无序不重复元素集合
    seen.add(s)
    while len(queue)>0:
        vertex=queue.pop(0)
        nodes=graph[vertex]
        for node in nodes:
            if node not in seen:
                queue.append(node)
                seen.add(node)
        result.append(vertex)
    return result