```python
import matplotlib.pyplot as plt
# ^^^ pyforest auto-imports - don't write above this line
import numpy as np
```


```python
x_1 = np.linspace(120000,150000, 1000000)
y_1 = (16*x_1**3/3 + 20*x_1/3+4)*np.exp(-0.0003125*x_1)
plt.plot(x_1,y_1)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



    
![png](Exercise%203.B_files/Exercise%203.B_1_2.png)
    



```python
x_1[np.argmin(y_1[y_1 >= 0.1])]
```




    125423.73542373543




```python

```
