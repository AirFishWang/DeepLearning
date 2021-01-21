https://github.com/yangsuhui/pytorch-handbook


我的笔记：
model.train(): 将模型设置成train模式 启用 BatchNormalization 和 Dropout
model.eval()：将模型设置成test模式，仅仅当模型中有Dropout和BatchNorm是才会有影响；
model.float()：将模型数据类型转换为float；
model.half()：将模型数据类型转换为half；
