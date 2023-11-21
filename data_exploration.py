import pandas as pd
from matplotlib import pyplot as plt
from data_preprocessing import read_data

data = read_data()

data['order_day'] = pd.to_datetime(data['order_day'])
data.sort_values(by='order_day', ascending=True, inplace=True)
data['year_quarter'] = data['order_day'].dt.to_period('Q')

colors = {
    'train': 'b',
    'test': 'r'
}
    
# quarterly call volume chart
ax = data.groupby(
    by='year_quarter'
).size().plot(
    kind='bar',
    title='Quarterly Call Volume',
    xlabel='Year Quarter',
    ylabel='Call Volume',
    color=[colors[i] for i in data.groupby(by='year_quarter')['set'].head(1)]
)
ax.set_xticks(ax.get_xticks()[::2])
labels = data['set'].unique()
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]
plt.legend(handles, labels, title='Dataset')
plt.tight_layout()
plt.savefig('tables_charts/quarterly_call_volume.png')
plt.clf()

# quarterly EcoShare conversions
ax = data.groupby(by='year_quarter')['accept'].sum().plot(
    kind='bar',
    title='Quarterly EcoShare Conversions',
    xlabel='Year Quarter',
    ylabel='Conversions',
    color=[colors[i] for i in data.groupby(by='year_quarter')['set'].head(1)]
)
ax.set_xticks(ax.get_xticks()[::2])
labels = data['set'].unique()
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]
plt.legend(handles, labels, title='Dataset')
plt.tight_layout()
plt.savefig('tables_charts/quarterly_conversions.png')
