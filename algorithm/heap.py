from typing import List


def heapify(nums: List[int], heap_size: int, root: int) -> None:
    largest = root

    left = 2 * root + 1
    right = 2 * root + 2

    if left < heap_size and nums[left] > nums[largest]:
        largest = left

    if right < heap_size and nums[right] > nums[largest]:
        largest = right

    if largest != root:
        nums[root], nums[largest] = nums[largest], nums[root]

        heapify(nums, heap_size, largest)


def heap_sort(nums: List[int]) -> List[int]:
    n = len(nums)

    # 建立大顶堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(nums, n, i)

    # 将最大值依次放到列表末尾
    for end in range(n - 1, 0, -1):
        nums[0], nums[end] = nums[end], nums[0]
        heapify(nums, end, 0)

    return nums

if __name__ == "__main__":
    numbers = [5, 3, 8, 2, 1, 7]

    print("排序前：", numbers)

    result = heap_sort(numbers)

    print("排序后：", result)