import plotly.express as px

data = dict(
    character=["家庭攒钱账户", "股票类", "债券类", "商品类", "发达市场", "新兴市场", "国内债", "国内商品",
               "美股（标普500）","A股（红利低波）","港股（港股通高股息）","中债新综合","黄金ETF",],
    parent=["", "家庭攒钱账户", "家庭攒钱账户", "家庭攒钱账户", "股票类", "股票类", "债券类", "商品类", "发达市场",
            "新兴市场","新兴市场","国内债","国内商品"],
    value=[100, 60, 30, 10, 30, 30, 30, 10, 30, 15, 15, 30, 10])

fig = px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
    branchvalues="total",
)

fig.update_traces(textinfo='label+percent root')
fig.show()
