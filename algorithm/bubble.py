from typing import List


def bubble_sort(nums:List[int])->List[int]:
    for i in range(len(nums)-1):
        swapped=False
        for j in range(len(nums)-1-i):
            if nums[j]>nums[j+1]:
                nums[j],nums[j+1]=nums[j+1],nums[j]
                swapped=True
        if not swapped:
            break
    return nums


if __name__=="__main__":
    numbers=[5, 3, 8, 2, 1, 7]
    print("排序前：", numbers)
    result = bubble_sort(numbers)
    print("排序后：", result)