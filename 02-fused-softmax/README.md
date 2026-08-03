# Softmax 简介

Softmax 在计算时，会用到如下公式：
$$ softmax(x_i) = \frac{e^{x_i}}{\Sigma^N_{i=1} e^{x_i}} $$

为了避免极大值带来的数值溢出，一般会采用如下的方式：

$$ softmax(x_i) = \frac{e^{x_i - max(x)}}{\Sigma^N_{i=1} e^{x_i - max(x)}} $$

