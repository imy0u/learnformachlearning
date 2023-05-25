'''计数排序
值等于数组下标，数组记录个数，
然后根据这个数组将元素进行正确的排序
'''
def countingSort(nums):
    bucket=[0]*(max(nums)+1)
    for num in nums:
        bucket[num]+=1
    i =0
    for j in range(len(bucket)):
        while bucket[j]>0:
            nums[i]=j
            bucket[j]-=1
            i+=1
    return nums
nums=[1,3,2,3,4,5,6,3,5,2,4,1]
print(countingSort(nums))