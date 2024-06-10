#1-reverse
"""
def reverse(num):
  if num == 0:
    return 0
  else:
    return (num % 10) * 10 + reverse(num // 10)
number = 1234
reversed_number = reverse(number)
print(f"The reverse of {number} is {reversed_number}")
"""
#2-perfect no
"""
def is_perfect(num):
  if num <= 1:
    return False
  divisor_sum = 1
  for i in range(2, int(num**0.5) + 1):
    if num % i == 0:
      divisor_sum += i
      if i * i != num:
        divisor_sum += num // i
  return divisor_sum == num
number = 6
if is_perfect(number):
  print(f"{number} is a perfect number")
else:
  print(f"{number} is not a perfect number")
"""

#3
"""
def print_array(arr):
  for element in arr:
    print(element, end=" ")
  print()
def linear_search(arr, target):
  for i, element in enumerate(arr):
    if element == target:
      return i
  return -1
# Example usage
arr = [10, 20, 30, 40, 50]
target = 30
print("Array:", end=" ")
print_array(arr)
index = linear_search(arr, target)
if index != -1:
  print(f"Target {target} found at index {index}")
else:
  print(f"Target {target} not found in the array")
"""
#4-Linear search-Non recursive
"""
def linear_search(arr,tar):
  for i in range(len(arr)):
    if arr[i]==tar:
      return i
  return -1
arr=[1,2,3,4,5,6,7,8,9]
tar=5
res=linear_search(arr,tar)
if res!= -1:
  print("Element found at index",res)
else:
  print("Element not found")
#Recursive-Factorial
def fac(n):
  if n==0:
    return 1
  else:
    return n*fac(n-1)
n=5
res=fac(n)
print("Factorial of",n,"is",res)
"""
#5-Master thm
"""
from math import log2
def master_theorem(a, b, f_n):
    if a < b**f_n(1):
        return "O(" + str(f_n(1)) + ")"
    elif a == b**f_n(1):
        return "O(" + str(f_n(1)) + " log n)"
    else:
        return "O(" + str(a) + "^n)"
a = 2
b = 2
f_n = lambda n: n 
print(master_theorem(a, b, f_n)) 
def substitution_method(T_n, guess):
    n = 1
    while True:
        if T_n(n) == guess(n):
            n *= 2
        else:
            break
        return "O(" + str(guess(1)) + ")"
T_n = lambda n: 2*T_n(n/2) + n 
guess = lambda n: n*log2(n)
print(substitution_method(T_n, guess)) 
def iteration_method(T_n):
    n = 1
    iterations = 0
    while True:
        if T_n(n) == 1:
            break
        n *= 2
        iterations += 1
        return "O(" + str(2**iterations) + ")"
T_n = lambda n: 2*T_n(n/2) + n 
print(iteration_method(T_n))
"""
#6-int array
"""
n1 = [1,2,3,4]
n2 = [3,4,5,6]
for i in n1:
  for j in range(i+1):
    if n1[i]==n2[j]:
      print(n1[i])
"""
#7
"""
def intersect(nums1, nums2):
  count1 = {}
  count2 = {}
  for num in nums1:
    if num in count1:
      count1[num] += 1
    else:
      count1[num] = 1
      for num in nums2:
        if num in count2:
          count2[num] += 1
        else:
          count2[num] = 1
  result = []
  for num in count1:
    if num in count2:
      result.extend([num] * min(count1[num], count2[num]))
  return result
nums1 = [1, 2, 2, 1]
nums2 = [2, 2]
print(intersect(nums1, nums2)) 
nums1 = [4, 9, 5]
nums2 = [9, 4, 9, 8, 4]
print(intersect(nums1, nums2))  
 """
#8
"""
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    return merge(left, right)
def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
result.extend(left[i:])
result.extend(right[j:])
return result
def sort_array(nums):
    return merge_sort(nums)
nums = [5, 2, 8, 3, 1, 6, 4]
print(sort_array(nums))  
 """ 
#9
"""
ar = []
e = []
o = []
for i in range(1, 11):
    ar.append(i)
ar.sort()
for num in ar:
    if num % 2 == 0:
        e.append(num)
    else:
        o.append(num)
print("Odd numbers", o)
print("even number", e)
 """ 

