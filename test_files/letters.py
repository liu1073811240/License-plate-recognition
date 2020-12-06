letters = [chr(x + ord('A')) for x in range(26) if not chr(x + ord('A')) in ['I', 'O']]

a = [chr(x + ord('A')) for x in range(26)]
print(a)  # ['A', 'B', 'C', 'D',...]
print(ord('A'))  # 65

print(letters)

digits = ['{}'.format(x + 1) for x in range(9)] + ['0']
print(digits)  # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

print(letters + digits)