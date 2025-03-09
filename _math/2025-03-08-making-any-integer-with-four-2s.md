---
layout: post
title: 如何通过4个2构造任意整数
date: 2025-03-08 22:20:00
description: 介绍一种方法，通过4个2构造任意整数
tags: 杂项 转载
categories: math
---

这篇文章是我从[YC](https://news.ycombinator.com/)上的帖子中看到的，相当于一个转载，感兴趣请参考[原始博客](https://eli.thegreenplace.net/2025/making-any-integer-with-four-2s/).

有这样一个数学迷题：
> 给定4个数字 2 和某个目标自然数，使用任何数学运算用这些 2 生成目标数字，不使用其他数字.

一些简单的例子如下:

$$
\begin{aligned}
& 1=\frac{2+2}{2+2} \\
& 2=\frac{2}{2}+\frac{2}{2} \\
& 3=2 \cdot 2-\frac{2}{2} \\
& 4=2+2+2-2 \\
& 5=2 \cdot 2+\frac{2}{2} \\
& 6=2 \cdot 2 \cdot 2-2
\end{aligned}
$$

在进一步：

$$
\begin{aligned}
18 & =2^{2^2}+2 \\
28 & =(2+2)!+2+2 \\
256 & =(2+2)^{2+2} \\
65536 & =2^{2^{2^2}}
\end{aligned}
$$

再有就是利用类似22(有两个2)这样的数来构造：

$$
\begin{aligned}
26 & =22+2+2 \\
11 & =\frac{22}{\sqrt{2+2}} \\
444 & =222 \cdot 2
\end{aligned}
$$

但是构造 7 是出了名的困难，但如果你允许使用更多数学工具，如 [Gamma 函数](https://en.wikipedia.org/wiki/Gamma_function)，就会变得容易。

$$
7=\Gamma(2)+2+2+2
$$

你的数学技能越高，他们能做出的数字就越多。可以参考这个[帖子](https://math.stackexchange.com/questions/1034122/get-the-numbers-from-0-30-by-using-the-number-2-four-times)，了解使用积分、循环分数和组合运算符的一些有趣组合。比如涉及复数的一个例子:

$$
12=|2+2 \sqrt{-2}|^2
$$

事实上，这似乎是 20 世纪 20 年代数学家最喜欢的消遣方式。直到[保罗·狄拉克 (Paul Dirac)](https://en.wikipedia.org/wiki/Paul_Dirac) 为每个数字找到了一个通用解，毁掉了所有人的乐趣。

原理就是嵌套的平方根：

$$
\begin{aligned}
\sqrt{2} & =2^{\frac{1}{2}}=2^{2^{-1}} \\
\sqrt{\sqrt{2}} & =2^{\frac{1}{4}}=2^{2^{-2}} \\
\sqrt{\sqrt{\sqrt{2}}} & =2^{\frac{1}{8}}=2^{2^{-3}}
\end{aligned}
$$

如果嵌套n次：

$$
\sqrt{\sqrt{\cdots n \cdots \sqrt{2}}}=2^{2^{(-n)}}
$$

然后得到：

$$
\log _2 2^{2^{(-n)}}=2^{(-n)}
$$

推出:

$$
\log _2\left(\log _2 2^{2^{(-n)}}\right)=-n
$$

$$
n=-\log _2\left(\log _2(\sqrt{\sqrt{\cdots n \cdots \sqrt{2}}})\right)
$$

但是你会发现，它使用了三个数字 2，而不是4个。不过，这很容易修改；因为 ，我们可以用它替换任何单个数字，得到4个:

$$
n=-\log _{\sqrt{2+2}}\left(\log _2(\sqrt{\sqrt{\cdots n \cdots \sqrt{2}}})\right)
$$

所以7就可以用如下方式构造：

$$
7=-\log _{\sqrt{2+2}}\left(\log _2(\sqrt{\sqrt{\sqrt{\sqrt{\sqrt{\sqrt{\sqrt{2}}}}}}})\right)
$$
