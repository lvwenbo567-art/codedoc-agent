from typing import List
def partition(nums:List[int],start:int,end:int)->int:
    tmp=nums[start]
    while start<end:
        while nums[end]>=tmp and start<end:
            end=end-1
        nums[start]=nums[end]
        while nums[start]<=tmp and start<end:
            start=start+1
        nums[end]=nums[start]
    nums[start]=tmp
    return start

def quick_sort(nums:List[int],start:int,end:int)->List[int]:
    if start<end:
        mid=partition(nums,start,end)
        quick_sort(nums,start,mid-1)
        quick_sort(nums,mid+1,end)
    return nums

if __name__ == "__main__":
    numbers = [5, 3, 8, 2, 1, 7]

    print("排序前：", numbers)

    result = quick_sort(numbers,0,len(numbers)-1)

    print("排序后：", result)