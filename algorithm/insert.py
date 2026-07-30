from typing import List
def insertion(nums:List[int])->List[int]:
    n=len(nums)
    for i in range(1,n):
        current=nums[i]
        j=i-1
        while j>=0 and nums[j]>current:
            nums[j+1]=nums[j]
            j=j-1
        nums[j+1]=current
    return nums


if __name__ == "__main__":
    numbers = [5, 3, 8, 2, 1, 7]

    print("排序前：", numbers)

    result = insertion(numbers)

    print("排序后：", result)