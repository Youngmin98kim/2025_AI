import numpy as np
v = np.array([1,3.9,-9,2])
print(v,v.ndim)
#numpy의 기능? 내부 원소를 같은 type으로 만들도록 설계

import pandas as pd

df = pd.DataFrame ({"a":[4,5,6],"b":[7,8,9],"c":[10,11,12]}, index = [1,2,3]
 )

print(df)
#key : column 명칭

#2차원 리스트를 사용하는 방법
df = pd.DataFrame([[4,7,10],[5,8,11],[6,9,12]], index=[1,2,3],columns = ['a','b','c'])
print(df)