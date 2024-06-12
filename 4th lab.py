#1 counting elements
"""
ar =[1,1,3,3,5,5,7,7]
c=0
for i in range(len(ar)-1):
    if ar[i]+ ar[i+1] in ar:
        c+=2
print(c)
"""
#2 string shifts
"""
s = "abc"
shift = [[0, 1], [1, 2]]

def left(a, s):
    return s[a:] + s[:a]

def right(a, s):
    return s[-a:] + s[:-a]

while shift:
    for i in range(len(shift)):
        if shift[i][0] == 0:
            a = shift[i][1]
            s = left(a, s)
        else:
            b = shift[i][1]
            s = right(b, s)
    shift.pop(0)

print(s)
"""
#3 Leftmost Column with at Least a One
"""
class BinaryMatrix:
    def __init__(self, mat):
        self.mat = mat
    def get(self, row: int, col: int) -> int:
        return self.mat[row][col]
    def dimensions(self) -> list:
        return [len(self.mat), len(self.mat[0])]
def leftMostColumnWithOne(binaryMatrix):
    rows, cols = binaryMatrix.dimensions()
    current_row = 0
    current_col = cols - 1
    leftmost_col_with_one = -1   
    while current_row < rows and current_col >= 0:
        if binaryMatrix.get(current_row, current_col) == 1:
            leftmost_col_with_one = current_col
            current_col -= 1  
        else:
            current_row += 1  
    return leftmost_col_with_one
mat1 = [[0, 0], [1, 1]]
binaryMatrix1 = BinaryMatrix(mat1)
print(leftMostColumnWithOne(binaryMatrix1))  

mat2 = [[0, 0], [0, 1]]
binaryMatrix2 = BinaryMatrix(mat2)
print(leftMostColumnWithOne(binaryMatrix2))  
mat3 = [[0, 0], [0, 0]]
binaryMatrix3 = BinaryMatrix(mat3)
print(leftMostColumnWithOne(binaryMatrix3))
"""
#4 First unique no
"""
from collections import deque
class Queue:
    def __init__(self, nums):
        self.queue = deque(nums)
        self.unique_elements = set(nums)
    def showUnique(self):
        if self.unique_elements:
            return self.queue[0]
        return -1
    def add(self, value):
        if value in self.unique_elements:
            self.unique_elements.remove(value)
        else:
            self.queue.append(value)
            self.unique_elements.add(value)

        while self.queue and self.queue[0] not in self.unique_elements:
            self.queue.popleft()
s = ["FirstUnique", "showFirstUnique", "add", "showFirstUnique", "add", "showFirstUnique", "add", "showFirstUnique"]
ar = [[2, 3, 5], [], [5], [], [2], [], [3], []]
firstUnique = None
ans = []
for i, op in enumerate(s):
    if op == "FirstUnique":
        firstUnique = Queue(ar[i])
        ans.append(None)
    elif op == "showFirstUnique":
        ans.append(firstUnique.showUnique())
        ans.append(",null")
    elif op == "add":
        firstUnique.add(ar[i][0])
print(ans)
"""
#5
"""
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def construct_tree(lst):
    if not lst:
        return None
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    while i < len(lst):
        node = queue.pop(0)
        if lst[i] is not None:
            node.left = TreeNode(lst[i])
            queue.append(node.left)
        i += 1
        if i < len(lst) and lst[i] is not None:
            node.right = TreeNode(lst[i])
            queue.append(node.right)
        i += 1
    return root
def isValidSequence(root, arr):
    if not root or not arr:
        return False
    if root.val != arr[0]:
        return False
    if len(arr) == 1:
        return not root.left and not root.right
    return isValidSequence(root.left, arr[1:]) or isValidSequence(root.right, arr[1:])
lst = [0,1,0,0,1,0,None,None,1,0,0]
arr = [0,1,0,1]
root = construct_tree(lst)
print(isValidSequence(root, arr))  
"""
#6 Kids With the Greatest Number of Candies
"""
def kidsWithCandies(candies, extraCandies):
    max_candies = max(candies)
    result = [candy + extraCandies >= max_candies for candy in candies]
    return result
candies = [2,3,5,1,3]
extraCandies = 3
print(kidsWithCandies(candies, extraCandies))  
candies = [4,2,1,1,2]
extraCandies = 1
print(kidsWithCandies(candies, extraCandies))
candies = [12,1,12]
extraCandies = 10
print(kidsWithCandies(candies, extraCandies))  
"""
#7 Max Difference You Can Get From Changing an Integer
"""
def maximumGap(num: int) -> int:
    num_str = str(num)
    max_num = int(''.join('9' if c != '0' else c for c in num_str))
    min_num = int(''.join('1' if c == '9' else '0' if c == '0' else '1' for c in num_str))
    return max_num - min_num
num = 555
print(maximumGap(num))
"""
#8 Check If a String Can Break Another String
"""
def checkIfCanBreak(s1: str, s2: str) -> bool:
    s1_sorted = sorted(s1)
    s2_sorted = sorted(s2)

    return (all(x >= y for x, y in zip(s1_sorted, s2_sorted)) or
            all(x >= y for x, y in zip(s2_sorted, s1_sorted)))
s1 = "abc"
s2 = "xya"
print(checkIfCanBreak(s1, s2))  
"""
#9 Number of Ways to Wear Different Hats to Each Other
"""
MOD = 10 ** 9 + 7
def numberWays(hats):
    n = len(hats)
    max_hat = 40
    hat_to_people = [[] for _ in range(max_hat + 1)]

    for person, hat_list in enumerate(hats):
        for hat in hat_list:
            hat_to_people[hat].append(person)
    dp = [0] * (1 << n)
    dp[0] = 1
    for hat in range(1, max_hat + 1):
        for mask in range((1 << n) - 1, -1, -1):
            for person in hat_to_people[hat]:
                if mask & (1 << person) == 0:
                    dp[mask | (1 << person)] += dp[mask]
                    dp[mask | (1 << person)] %= MOD
    return dp[(1 << n) - 1]
hats = [[3, 4], [4, 5], [5]]
print(numberWays(hats))  
hats = [[3, 5, 1], [3, 5]]
print(numberWays(hats))  
hats = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
print(numberWays(hats))  
"""
#10 Destination City
"""
def destCity(paths):
    outgoing = set()
    for path in paths:
        outgoing.add(path[0])
    for path in paths:
        if path[1] not in outgoing:
            return path[1]
paths = [["London", "New York"], ["New York", "Lima"], ["Lima", "Sao Paulo"]]
print(destCity(paths)) 
paths = [["B", "C"], ["D", "B"], ["C", "A"]]
print(destCity(paths))  
paths = [["A", "Z"]]
print(destCity(paths))
"""
