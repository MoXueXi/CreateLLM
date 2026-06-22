# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

import re
text = "Hello, world. This is a -- test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item for item in result if item.strip()]
print(result)