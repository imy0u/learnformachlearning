'''
桶排序
工作原理是 将数组分到有限数量的桶里面
每个桶内再 个别排序'''
def bucketSort(nums,defaultBucketSize=5):
    maxVal,minVal=max(nums),min(nums)
    bucketSize=defaultBucketSize
    bucketCount=(maxVal-minVal)//bucketSize+1#数据分多少组
    buckets=[]
    for i in range(bucketCount):
        buckets.append([])#利用函数映射将各个数据放入对应的桶内
    for num in nums:
        buckets[(num-minVal)//bucketSize].append(num)
    nums.clear()
    for bucket in buckets:
        # insertionSort(bucket)#桶内使用的排序算法
        nums.extend(bucket)
    return nums