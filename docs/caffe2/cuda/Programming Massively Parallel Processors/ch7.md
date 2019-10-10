# Ch7
## Notes
* NAN representation?
## Solution
### 7.1
![table](./images/no_zero_chart.jpg)

![chart](./images/no_zero_axis.jpg)
### 7.2
![table](./images/7_2_chart.jpg)

![chart](./images/7_2_axis.jpg)

The number of E represents the range of representation.

In scope near __0__, it increases granularity.

In scope away __0__, it increase max number that could be represented.
### 7.3
``` c
NE = number of bits for E
NM = number of bits for M

f1 = first addition parameter
f2 = second addition parameter

e1 = E of f1
e2 = E of f2
m1 = M of f1
m2 = M of f2

sum = f1 + f2
eSum = E of sum
mSum = M of sum
```
Suppose e2 < e1.

1)
If |e1 - e2| = NM, there must be overflow as e2 would be shifted right by NM bits.

If the m2 = 1<sub>0</sub>1<sub>1</sub>...1<sub>NM</sub>, the error is (1 - 2<sup>-NM</sup>) ulp.

2)
If |e1 - e2| > NM, after shift, the m2 would be like 0<sub>0</sub>0<sub>1</sub>...0<sub>n</sub>1<sub>0</sub>1<sub>1</sub>...1<sub>NM-n</sub> < 1<sub>0</sub>1<sub>1</sub>...1<sub>NM</sub>

The round error is < (1 - 2<sup>-NM</sup>)

3)
If |e1 - e2| < NM, after shift m1, m2 left and then shift m2 right,
the tail to be rounded is
1<sub>0</sub>1<sub>1</sub>...1<sub>n</sub> < 1<sub>0</sub>1<sub>1</sub>...1<sub>NM</sub> (n < NM)

So the max is about 1 ulp
### 7.4
