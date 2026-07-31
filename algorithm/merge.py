from typing import List
def merge(left:List[int],right:List[int])->List[int]:
    i=0
    j=0
    result=[]
    while i<len(left) and j<len(right):
        if left[i]<=right[j]:
            result.append(left[i])
            i+=1
        else:
            result.append(right[j])
            j+=1
    while i<len(left):
        result.append(left[i])
        i+=1
    while j<len(right):
        result.append(right[j])
        j+=1
    return result


def merge_sort(nums:List[int])->List[int]:
    if len(nums)<=1:
        return nums
    mid=len(nums)//2
    left=merge_sort(nums[:mid])
    right=merge_sort(nums[mid:])
    return merge(left,right)


if __name__ == "__main__":
    numbers = [5, 3, 8, 2, 1, 7]

    print("排序前：", numbers)

    result = merge_sort(numbers)

    print("排序后：", result)
