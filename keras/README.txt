1. keras和tf.keras的一些差异性
(1) keras 和 tf.keras的初始化器的 __call__函数参数有差异，tf.keras的__call__函数多了一个partition_info参数
(2) keras的模型继承关系， 贯序模型继承于函数式模型， 函数式模型继承包装器Container，其中基类包装器中实现了函数load_weights, 函数式模型中没有重写load_weights, 贯序模型重写了load_weight函数
(3) tf.keras模型继承关系， 贯序模型继承于函数式模型， 函数式模型继承Network，其中类Network实现了函数load_weights, 函数式模型和贯序模型均没有重写load_weights函数
(4) tf.keras的load_weights函数没有skip_mismatch参数，若不匹配则直接报错， 而keras的load_weights函数有skip_mismatch参数，若不匹配，则continue
(5) tf.keras和keras多GPU训练问题
   相同点：首先都必须在CPU设备上构建single_model(模型的权值保存在内存中，而不是显存中), 然后调用multi_gpu_model函数在各个GPU上复制模型，返回parallel_model
          再对parallel_model进行编译(优化器设置)和训练
   i: keras对于多模型的保存： keras无法直接保存parallel_model的权值(其实各个GPU上parallel_model的权值是公用的，也就是single_model的权值)，所以在定义保存中间结果的回调对象时，
       需要使回调对象的model属性设置为single_model. 在retinanet源码中，通过类RedirectModel来实现这个过程(on_train_begin函数). 最终保存的模型是single_model
       由于是使用parallel_model来训练，使用single_model来保存， 因为single_model并没有经过编译，所以保存后的模型不会存储优化器的状态（single_model没有经过编译，其不存在optimizer属性）
       在模型保存的时候，save函数会检查model的optimizer属性，如果没有该属性则跳过保存optimizer属性，所以最终保存的模型只有模型的结构和权值，没有优化器的状态。

       保存的模型只有模型的结构和权值，没有优化器的状态，这样会导致一个问题。当训练意外终止时，从上一次中断的位置重新开始训练，由于我们没有优化器状态，所以必须再次编译模型，这样并不严格属于"继续训练"，因为
       我们丢失了上一次的学习率等等超参数。

   ii: tf.keras: 不同于keras， tf.keras的single_model没有进行编译，但是single_model仍然具有optimizer属性， 只不过是optimizer=None, tf.keras在保存模型时发现model具有了optimizer属性，于是
       调用optimizer.get_config()函数试图获取优化器配置，由于此时single_model的optimizer=None，所以这里会抛出错误。绕过此错误的方法：是对single_model进行编译，让其optimizer属性不为None,这样
       就可以保存模型了，需要注意的是这样保存的模型不仅存在权值，而且存在optimizer属性，但optimizer属性是没用用处的，只是为了绕过上述错误的方法。
2. 图片的测试结果(TP, TN, FP, FN)保存路径保存在bin下
3. 修改anchor的宽高比, 需要修改以下几处
   (1) utils/anchors.py中anchors_for_shape函数  
   (2) utils/anchors.py中generate_anchors函数
   (3) layers/_misc.py 类Anchors的构造函数
   (4) models/retinanet.py 中的AnchorParameters.default参数设置
4. 若需要求mAp， 程序只会保留高于阈值的检测框，则在执行evaluate_512.py的时候应该将分数阈值设置为零(或很低), 这样就可以记录各个阈值下的recall和precious，
   若需要看某一个阈值下，程序的检测结果，则需要将阈值设置为该值，并提供save_path, 用于保存图片检测结果(TP, TN, FP, FN)

train model:      python train_512.py --gpu=0,1 --multi-gpu-force csv
visualize model:  python show_model.py csv
evaluate model:   python evaluate.py --gpu=0
