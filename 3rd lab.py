#1
"""
ar=[1,1,3,3,5,5,7,7]
ele = set(ar)
c=0
for x in ar:
    if x+1 in ele:
        c+=1
print(c)
"""
#2
"""
s = "abc"
shift = [[0, 1], [1, 2]]
def left(a, s):
    return s[a:] + s[:a]
def right(a, s):
    return s[-a:] + s[:-a]
while len(shift):
    direction, amount = shift.pop(0)
    if direction == 0:
        s = left(amount, s)
    else:
        s = right(amount, s)
print(s)
"""
#3
"""
def get(matrix, row, col):
    return matrix[row][col]
def dimensions(matrix):
    return [len(matrix), len(matrix[0])]
def leftMostColumnWithOne(matrix):
    rows, cols = dimensions(matrix)
    current_row = 0
    current_col = cols - 1
    leftmost_col = -1
    while current_row < rows and current_col >= 0:
        if get(matrix, current_row, current_col) == 1:
            leftmost_col = current_col
            current_col -= 1
        else:
            current_row += 1
            return leftmost_col
        matrix = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0]
            ]
result = leftMostColumnWithOne(matrix)
print(result)
"""
#4
"""
from collections import deque, defaultdict
class FirstUnique:
    def __init__(self, nums):
        self.queue = deque()
        self.count = defaultdict(int)
        if isinstance(nums, list):
            for num in nums:
                self.add(num)
        else:
            self.add(nums)
    def showFirstUnique(self):
        while self.queue and self.count[self.queue[0]] > 1:
            self.queue.popleft()
            if self.queue:
                return self.queue[0]
            return -1
        def add(self, value):
            self.count[value] += 1
            if self.count[value] == 1:
                self.queue.append(value)
            else:
                while self.queue and self.count[self.queue[0]] > 1:
                    self.queue.popleft()
commands =["FirstUnique","showFirstUnique","add","showFirstUnique","add",
           "showFirstUnique","add","showFirstUnique"]
inputs = [[2,3,5],[],[5],[],[2],[],[3],[]]
first_unique = None
outputs = []
for cmd, vals in zip(commands, inputs):
    if cmd == "FirstUnique":
        first_unique = FirstUnique(vals)
        outputs.append(None)
    elif cmd == "showFirstUnique":
        outputs.append(first_unique.showFirstUnique())
    elif cmd == "add":
        first_unique.add(vals[0])
print(outputs)
"""
#5
"""
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def isValidSequence(root, arr):
    def dfs(node, index):
        if index == len(arr) - 1:
            return node is not None and node.val == arr[index] and
node.left is None and node.right is None
current value in arr, return False
if node is None or node.val != arr[index]:
return Fa lse
return dfs(node.left, index + 1) or dfs(node.right, index + 1)return dfs(root, 0)
root = TreeNode(0)
root.left = TreeNode(1)
root.right = TreeNode(0)
root.left.left = TreeNode(0)
root.left.right = TreeNode(1)
root.right.left = TreeNode(0)
root.right.right = None
root.left.left.left = None
root.left.left.right = None
root.left.right.left = None
root.left.right.right = TreeNode(1)
arr = [0, 1, 0, 1]
print(isValidSequence(root, arr))
"""
#6
"""
def kidsWithCandies(candies, extraCandies):
    max_candies = max(candies)
    result = []
    for candy in candies:
        result.append(candy + extraCandies >= max_candies)
    return result
print(kidsWithCandies([2, 3, 5, 1, 3], 3)) 
print(kidsWithCandies([4, 2, 1, 1, 2], 1))
print(kidsWithCandies([12, 1, 12], 10))
"""
#7
"""
def maxDiff(num):
    num_str = str(num)
    max_num = num_str[:]
    min_num = num_str[:]
    for i in range(10):
        max_num = max_num.replace(str(i), '9')
        if max_num != '0':
            break
        if min_num[0] != '1':
            min_num = min_num.replace(min_num[0], '1')
        else:
            for i in range(1, len(min_num)):
                if min_num[i] != '0' and min_num[i] != min_num[0]:
                    min_num = min_num.replace(min_num[i], '0')
                    break
        return int(max_num) - int(min_num)
print(maxDiff(555))
"""
#8
"""
def canBreak(s1, s2):
    s1_sorted = sorted(s1)
    s2_sorted = sorted(s2)
    can_break_s1 = all(s1_sorted[i] >= s2_sorted[i] for i in range(len(s1)))
    can_break_s2 = all(s2_sorted[i] >= s1_sorted[i] for i in range(len(s1)))
    return can_break_s1 or can_break_s2
print(canBreak("abc", "xya")) 
print(canBreak("abe", "acd")) 
"""
#9
MOD = 10 ** 9 + 7
def numberWays(hats):
    n = len(hats)
    num_hats = 0
    for h in hats:
        num_hats = max(num_hats, max(h))
        dp = [0] * (1 << (num_hats + 1))
        dp[0] = 1
    for i in range(n):
        for hat in hats[i]:
            for mask in range(1 << (num_hats + 1)):
                if mask & (1 << hat):
                    continuedp[mask | (1 << hat)] = (dp[mask | (1 << hat)] +
                                                     dp[mask]) % MOD
return dp[(1 << (num_hats + 1)) - 1]
print(numberWays([[3, 4], [4, 5], [5]])) # Output: 1
print(numberWays([[3, 5, 1], [3, 5]]))
