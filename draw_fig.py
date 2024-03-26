import matplotlib.pyplot as plt
import json

with open('../model/codevol-wizardcoder-0304-instructed-by-wizardcoder-reversed-problem-prompt-0318-cleaned/trainer_state.json') as f:
    data = json.load(f)

# 步骤 2：创建数据
x = [s['step'] for s in data['log_history'][:-1]]
y = [s['loss'] for s in data['log_history'][:-1]]

# 步骤 3：绘制折线图
plt.plot(x, y)

# 步骤 4：自定义图表样式（可选）
plt.xlabel('X轴标签')
plt.ylabel('Y轴标签')
plt.title('折线图示例')

# 步骤 5：添加标题、标签和图例（可选）
# 如果需要添加图例，可以使用 plt.legend()

# 步骤 6：显示图表或保存图表到文件
plt.savefig('loss.png')

