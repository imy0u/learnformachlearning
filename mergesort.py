'''
归并排序
采用分治法
将两个或两个以上的  有序表  合并成为一个新的有序表
'''
def mergeSort(nums):
    def merge(left,right):
        result=[]
        i=j=0
        while i<len(left) and j<len(right):
            if left[i]<=right[j]:
                result.append(left[i])
                i+=1
            else:
                result.append(right[j])
                j+=1
        result=result+left[i:]+right[j:]#将剩余元素直接添加到末尾
        return result
    if len(nums)<=1:
        return nums
    mid=len(nums)//2
    left=mergeSort(nums[:mid])
    right=mergeSort(nums[mid:])
    return merge(left,right)
nums=[12,21,31,13,41,14,135,53,32,643,23,63,32,435,23,523]
print(mergeSort(nums))