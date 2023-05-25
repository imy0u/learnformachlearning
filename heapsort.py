'''
堆排序
堆排序利用堆这种数据结构
堆是一个近似完全二叉树的结构
'''
def heapSort(nums):
    def adjustHeap(nums,i,size):
        lchild=2*i+1#这里按照完全二叉树的性质来进行定义
        rchild=2*i+2
        largest=i
        if lchild<size and nums[lchild]>nums[largest]:
            largest=lchild
        if rchild<size and nums[rchild]>nums[largest]:
            largest=rchild
        if largest!=i:
            nums[largest],nums[i]=nums[i],nums[largest]
            adjustHeap(nums,largest,size)
    
    def builtheap(nums,size):
        for i in range(len(nums)//2)[::-1]:
            adjustHeap(nums,i,size)
    size=len(nums)
    builtheap(nums,size)
    for i in range(len(nums))[::-1]:
        nums[0],nums[i]=nums[i],nums[0]
        adjustHeap(nums,0,i)
    return nums