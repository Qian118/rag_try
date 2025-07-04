from typing import List


def removeElement(nums: List[int], val: int) -> int:
    fast = 0  # 快指针
    slow = 0  # 慢指针
    size = len(nums)
    while fast < size:  # 不加等于是因为，a = size 时，nums[a] 会越界
        # slow 用来收集不等于 val 的值，如果 fast 对应值不等于 val，则把它与 slow 替换
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    return slow


nums=[3,2,2,3]
val=3
res=removeElement(nums,val)
print(removeElement(nums,val))