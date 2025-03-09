---
layout: post
title: c++中的move 和 forward
date: 2025-03-08 17:10:00
description: 介绍c++中的move和forward的区别及各自的实现方式。
tags: c++
categories: programming
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

## 一、左值和右值的区别

直观意义上 左值是可以被赋值的，在内存中有独立的地址

具体区别参考：

```
https://www.cnblogs.com/gqtcgq/p/9828247.html

https://blog.csdn.net/zpznba/article/details/86028607
```

## 二、 std::move

函数原型定义：

```c++
template<typename T> // in namespace std
typename remove_reference<T>::type&&
move(T&& param)
{
using ReturnType = // alias declaration;
typename remove_reference<T>::type&&; // see Item 9
return static_cast<ReturnType>(param);
}
```

所以看出std::move只是对参数类型做了一个强制转换，转换为右值引用类型，std::move并不能移动任何东西，它唯一的功能是将一个左值强制转化为右值引用，继而可以通过右值引用使用该值，以用于移动语义。

原型中的函数参数 T&& 是一个指向模板类型参数的右值引用，通过引用折叠，此参数可以与任何类型的实参匹配（可以传递左值或右值，这是std::move主要使用的两种场景）。关于引用折叠如下：

方式一、

X& &、X&& &、X& &&都折叠成X&，用于处理左值

```c++
string s("hello");
std::move(s) => std::move(string& &&) => 折叠后 std::move(string& )
此时：T的类型为string&
typename remove_reference<T>::type为string
整个std::move被实例化如下
string&& move(string& t) //t为左值，移动后不能在使用t
{
//通过static_cast将string&强制转换为string&&
return static_cast<string&&>(t);
}
```

方式二、

X&& &&折叠成X&&，用于处理右值

```c++
std::move(string("hello")) => std::move(string&&)
//此时：T的类型为string
//  remove_reference<T>::type为string
//整个std::move被实例如下
string&& move(string&& t) //t为右值
{
   return static_cast<string&&>(t);  //返回一个右值引用
}
```

简单来说，右值经过T&&传递类型保持不变还是右值，而左值经过T&&变为普通的左值引用.

参考

```
 https://blog.csdn.net/p942005405/article/details/84644069

《Effective.Modern.C++-42.Specific.Ways.to.Improve.Your.Use.of.C++11.and.C++14-2014》 Item23
```

## 三、 std::forward

大致原型为：

```c++
template< typename T > inline
T&& forward( typename std::remove_reference<T>::type& t ) noexcept {
    return (static_cast<T&&>(t));
}

template< typename T > inline
T&& forward( typename std::remove_reference<T>::type&& t ) noexcept {
    static_assert(!is_lvalue_reference<T>::value, "bad forward call");
    return (static_cast<T&&>(t));
}
```

std::forward()实现完美转发，意思是当传入参数是左值时返回左值引用，传入参数是右值时返回右值引用。

- std::move执行到右值的无条件转换。就其本身而言，它没有move任何东西。
- std::forward只有在它的参数绑定到一个右值上的时候，它才转换它的参数到一个右值。
- std::move和std::forward只不过就是执行类型转换的两个函数；std::move没有move任何东西，std::forward没有转发任何东西。在运行期，它们没有做任何事情。它们没有产生需要执行的代码，一byte都没有。
- std::forward()不仅可以保持左值或者右值不变，同时还可以保持const、Lreference、Rreference、validate等属性不变；
  一个输出的例子为：

```c++
#include <iostream>
#include <memory>
#include <utility>

struct A {
    A(int&& n) { std::cout << "rvalue overload, n=" << n << "\n"; }
    A(int& n)  { std::cout << "lvalue overload, n=" << n << "\n"; }
};

class B {
public:
    template<class T1, class T2, class T3>
    B(T1&& t1, T2&& t2, T3&& t3) :
        a1_{std::forward<T1>(t1)},
        a2_{std::forward<T2>(t2)},
        a3_{std::forward<T3>(t3)}
    {
    }

private:
    A a1_, a2_, a3_;
};

template<class T, class U>
std::unique_ptr<T> make_unique1(U&& u)
{
    return std::unique_ptr<T>(new T(std::forward<U>(u)));
}

template<class T, class... U>
std::unique_ptr<T> make_unique2(U&&... u)
{
    return std::unique_ptr<T>(new T(std::forward<U>(u)...));
}

int main()
{
    auto p1 = make_unique1<A>(2); // rvalue
    int i = 1;
    auto p2 = make_unique1<A>(i); // lvalue

    std::cout << "B\n";
    auto t = make_unique2<B>(2, i, 3);
}

Output:

rvalue overload, n=2
lvalue overload, n=1
B
rvalue overload, n=2
lvalue overload, n=1
rvalue overload, n=3

```

参考

```
https://blog.csdn.net/guoxiaojie_415/article/details/79902278

https://www.cnblogs.com/boydfd/p/5182743.html

https://eli.thegreenplace.net/2014/perfect-forwarding-and-universal-references-in-c/
```
