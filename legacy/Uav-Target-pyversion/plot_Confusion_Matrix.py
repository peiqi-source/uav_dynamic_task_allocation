from graphviz import Digraph

dot = Digraph(format='png')

dot.node('A', '开始', shape='circle')
dot.node('B', '输入目标点数据')
dot.node('C', '初始化簇中心和粒子群')
dot.node('D', '计算适应度')
dot.node('E', '更新粒子位置和速度')
dot.node('F', '判断收敛条件?', shape='diamond')
dot.node('G', '输出最终分簇结果')
dot.node('H', '目标点分配')
dot.node('I', '更新簇中心')
dot.node('J', '结束', shape='circle')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FH'])
dot.edge('F', 'G', label='是')
dot.edge('F', 'H', label='否')
dot.edge('H', 'I')
dot.edge('I', 'D')
dot.edge('G', 'J')

dot.render('pso_clustering_process')
