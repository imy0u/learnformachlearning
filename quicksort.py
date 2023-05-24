'''
快速排序
冒泡排序的一种改进算法
'''
def quickSort(nums):
    if len(nums)<=1:
        return nums
    pivot=nums[0]#第一轮的基准值
    left=[nums[i] for i in range(1,len(nums)) if nums[i]<pivot]
    right=[nums[i] for i in range(1,len(nums)) if nums[i]>=pivot]
    return quickSort(left)+[pivot]+quickSort(right)