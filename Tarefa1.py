S = ""
S = ")("
S = ")()"
S = "()[]"
S = "[{()}]"
array = []
for s in S:
    if s in ["{", "(", "["]:
        array.append(s)
    if s in ["}", ")", "]"] and array == []:
        array.append(s)
        break
    if s == ")" and array[-1] == "(":
        array.pop()
    if s == "]" and array[-1] == "[":
        array.pop()
    if s == "}" and array[-1] == "{":
        array.pop()
if array == []:
      print(1)
else:
    print(0)