'''
基数排序
基数排序是一种非比较型整数排序废，将整数按位数切割成不同的数字
然后按照每个位数分别进行比较
'''
def radixSort(nums):
    mod=10
    div=1
    mostBit=len(str(max(nums)))
    buckets=[[] for row in range(mod)]#构造mod个数列存放
    while mostBit:
        for num in nums:
            buckets[num//div%mod].append(num)#计算位置，并插入
        i=0 #每次排序完之后，要输出
        for bucket in buckets:
            while bucket:
                nums[i]=bucket.pop(0)#输出
                i+=1
        div*=10#除以的数，每次进一位
        mostBit-=1
    return nums