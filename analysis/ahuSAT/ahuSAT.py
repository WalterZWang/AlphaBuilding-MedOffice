import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_name = 'LR_shrink'

# read in the benchmark data
benchmark = 'in.csv'

benchmark = pd.read_csv(benchmark,usecols=['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)',
                                       'NODE 149:System Node Temperature [C](TimeStep)'])
benchmark.index = pd.date_range(start='1/1/2018', periods=len(benchmark), freq='1H')
benchmark.columns = ['Outdoor temp','Guideline 36']

# read in il data
il_file = file_name+'il.csv'
il = pd.read_csv(il_file, index_col=0)
il = il.iloc[:,[0,-1]]
il.index = pd.date_range(start='1/1/2018', periods=len(il),freq='15T')
il = il.resample('H').mean()
il.columns = ['Initialization','IL completed']

# read in rl data
rl_file = file_name+'_ddpg.csv'
ddpg_columns = ['2','10']
rl = pd.read_csv(rl_file, index_col=0)
rl = rl[ddpg_columns]
rl.index = pd.date_range(start='1/1/2018', periods=len(rl),freq='15T')
rl = rl.resample('H').mean()
rl.columns = ['RL epoch {0}'.format(i) for i in ddpg_columns]

# merge data
data = pd.concat([benchmark,il,rl],axis=1)
data.head()

# plot
summer = data.truncate(before='2018-07-16',after='2018-07-23')
summer.plot()
plt.title('Summer Week: AHU Supply Air Temp.', fontsize=18)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

winter = data.truncate(before='2018-01-16',after='2018-01-23')
winter.plot()
plt.title('Winter Week: AHU Supply Air Temp.', fontsize=18)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))