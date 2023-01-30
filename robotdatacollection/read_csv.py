import pandas as pd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./ep1.csv')
start_idx = df.index[df['Case'].diff() == 1].tolist()
stop_idx = df.index[df['Case'].diff() == -1].tolist()

            # df2 = df_new.iloc[start_idx[0]-buffer:stop_idx[0]+buffer]




plt.figure()
ax1 = plt.subplot(311)
ax1.plot(df.t.values, df.Fx.values, label='Fx')
ax1.legend()
ax1.grid()
ax1.set_ylabel('t')
ax1.set_xlabel('Fx')
ax1.set_title('Fx')


ax2 = plt.subplot(312)
ax2.plot(df.t.values, df.Fy.values, label='Fy')
ax2.legend()
ax2.grid()
ax2.set_title('Fy')

ax3 = plt.subplot(313)
ax3.plot(df.t.values, df.Fz.values, label='Fz')
ax3.legend()
ax3.grid()
ax3.set_title('Fz')

plt.figure()
plt.scatter(df.t.values, df.Case.values, label='Label')
plt.grid()


plt.show()

