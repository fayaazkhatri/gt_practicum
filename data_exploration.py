import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_excel(
    io='data/ecoshare_sales_v3.xlsx',
    sheet_name='Data',
    header=0,
    true_values=['Y'],
    false_values=['N']
)

data['order_day'] = pd.to_datetime(data['order_day'])
data.sort_values(by='order_day', ascending=True, inplace=True)
data['year_month'] = data['order_day'].dt.strftime('%Y-%m')

# monthly call volume chart
ax = data.groupby(
    by='year_month'
).size().plot(
    kind='bar',
    title='Monthly Call Volume',
    xlabel='Year-Month',
    ylabel='Call Volume'
)
ax.set_xticks(ax.get_xticks()[::2])
plt.tight_layout()
plt.savefig('tables_charts/monthly_call_volume.png')
plt.clf()

# monthly EcoShare conversions
ax = data.groupby(by='year_month')['accept'].sum().plot(
    kind='bar',
    title='Monthly EcoShare Conversions',
    xlabel='Year-Month',
    ylabel='Conversions'
)
ax.set_xticks(ax.get_xticks()[::2])
plt.tight_layout()
plt.savefig('tables_charts/monthly_conversions.png')
