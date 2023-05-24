'''
插入排序
基本思想是：将一个元素插入已经排好序的有序表中
'''
def insertSort(nums):
    for i in range(len(nums)-1):
        curNum,preIndex=nums[i+1],i#保存当前待插入的数
        while preIndex>=0 and curNum<nums[preIndex]:
            nums[preIndex+1]=nums[preIndex]#元素后移
            preIndex-=1
        nums[preIndex+1]=curNum
    return nums
nums=[213,34,57,53,65,3,25,6,3,1,15,9,45,89]
print(insertSort(nums))