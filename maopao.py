'''
排序算法：冒泡排序
冒泡排序一定是最大的在最后边，每次都产生一个单位内的最大值，所以，内循环每次少一
'''
def bubblesort(nums):
    for i in range(len(nums)-1):
        for j in range(len(nums)-i-1):
            if nums[j]>nums[j+1]:
                nums[j],nums[j+1]=nums[j+1],nums[j]
    return nums
nums=[4,5,21,6,2,4,5,7,9,1,4,8]
print(bubblesort(nums))