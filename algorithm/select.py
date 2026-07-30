from typing import List


def selection_sort(nums: List[int]) -> List[int]:
    n=len(nums)
    for i in range(n-1):
        min=i
        for j in range(i+1,n):
            if nums[j]<nums[min]:
                min=j
        nums[i],nums[min]=nums[min],nums[i]
    return nums

if __name__ == "__main__":
    numbers = [5, 3, 8, 2, 1, 7]

    print("排序前：", numbers)

    result = selection_sort(numbers)

    print("排序后：", result)