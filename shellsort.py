'''希尔排序
插入排序的改进版本，递减增量排序算法。
先将整个待排序记录序列分隔成为若干子序列分别进行直接插入排序，待整个序列中的记录基本有序时
再对全体记录进行一次直接插入排序'''
def shellSort(nums):
    lens=len(nums)
    print('长度为:'+str(lens))
    gap=lens//2

    while gap>0:
        print('gap目前大小是：'+str(gap))
        for i in range(gap,lens):
            curNum,preIndex=nums[i],i-gap#curNum保存当前待插入的数
            print('curNum的大小是'+str(curNum))
            print('preIndex的大小是'+str(preIndex))
            print('当前nums'+str(nums))
            while preIndex>=0 and curNum<nums[preIndex]:
                nums[preIndex+gap]=nums[preIndex]
                preIndex-=gap
            nums[preIndex+gap]=curNum
        gap//=2#向左取整
    return nums
nums=[12,21,34,43,51,15,16,64,73,26,29]
shellSort(nums)
print(nums)