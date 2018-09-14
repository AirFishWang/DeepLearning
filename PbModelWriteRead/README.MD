关于模型固化有两种方式, 详细内容可以查看下面链接

https://www.jianshu.com/p/091415b114e2

本程序使用write_graph函数来固化不带变量的图（程序中将变量变成了常量）, 一般情况下不这么做,好的做法可以参考../tensorrt_site_package/examples/tf_to_trt/lenet5.py
的固化过程