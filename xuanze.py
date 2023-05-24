'''
选择排序
将剩余待排序的元素中选出最小或最大的那个，跟当前待排序中元素最前面的那个对换位置
'''
def selectionSort(nums):
    for i in range(len(nums)-1):
        minmin=i
        for j in range(i+1,len(nums)):
            if nums[j]<nums[minmin]:
                minmin=j
        nums[i],nums[minmin]=nums[minmin],nums[i]
    return nums
nums=[8,5,13,7,3,6,4,5,93,63,3,0,73,6,43,23]
print(selectionSort(nums))