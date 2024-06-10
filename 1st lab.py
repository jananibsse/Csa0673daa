#1-Fibonacci
"""
def fibo(n):
    if n<=1:
        return n
    else:
        return (fibo(n-1)+(n-2))
n=10
if n<=0:
    print("Please enter a positive integer:")
else:
    print("Fibonacci sequence: ")
    for i in range(n):
        print(fibo(i))
"""
#2-Armstrong no
"""
def arms(num,n1,sum,temp):
    if temp==0:
        if sum==num:
            return True
        else:
            return False
    digit=temp%10
    sum=sum+digit**n1
    temp=temp//10
    return arms(num,n1,sum,temp)
num=int(input("enter no"))
sum=0
n1=len(str(num))
temp=num
res=arms(num,n1,sum,temp)
if res:
    print(num,"armstrong no")
else:
    print(num,"not armstrong no")
    """
#3-gcd
"""
def gcd(a,b):
    if (b==0):
        return a
    else:
        return gcd(b,a%b)
a=int(input("enter a:"))
b=int(input("enter b:"))
GCD=gcd(a,b)
print("GCD:",GCD)
"""
#4-Largest element
"""
def largest(arr, n):
	max = arr[0]
	for i in range(1, n):
		if arr[i] > max:
			max = arr[i]
	return max
arr = [10, 324, 45, 90, 9808]
n = len(arr)
Ans = largest(arr, n)
print("Largest in given array ", Ans)
"""
#5-Factorial
"""
def recur_fact(n):
   if n == 1:
       return n
   else:
       return n*recur_fact(n-1)
num = 5
if num < 0:
   print("Factorial does not exist for negative numbers")
elif num == 0:
   print("The factorial of 0 is 1")
else:
    print("The factorial of", num, "is", recur_fact(num))
"""
#6-Copy string
"""
def copy_string(source, destination=""):
  if not source:
    return destination
  return copy_string(source[1:], destination + source[0])
source_string = "Hello, world!"
destination_string = copy_string(source_string)
print(destination_string)
"""
#7-Reverse string
"""
def rev_str(t):
  if not t:
    return
  rev_str(t[1:])
  print(t[0], end="")
t = "Hello, world!"
rev_str(t)
"""
#8&9 -prime
"""
def is_prime(num):
  if num <= 1:
    return False
  for i in range(2, int(num**0.5) + 1):
    if num % i == 0:
      return False
  return True
prime_count = 0
num = 2
while prime_count < 10:
  if is_prime(num):
    print(num)
    prime_count += 1
  num += 1
  """
#10- String Palindrome
"""
def is_palindrome(s):
    if len(s) <= 1:
        return True
    if s[0] == s[-1]:
        return is_palindrome(s[1:-1])
    return False
test_strings = ["radar", "hello", "level", "world", "madam"]
for string in test_strings:
    print(f"'{string}' is a palindrome: {is_palindrome(string)}")
    """

