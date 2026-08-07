---
layout: post
title: c++中的 move 和 forward
date: 2025-03-08 17:10:00
description: 介绍c++中的move和forward的区别及各自的实现方式。
tags: [C++, 现代C++, 移动语义]
categories: [C++ 语言]
giscus_comments: false
related_posts: false
toc:
  sidebar: left
---

## 一、左值和右值的区别

直观意义上理解，左值是可以被赋值的，在内存中有独立的地址的值。左值在表达式结束后依然存在的、有持久身份的对象或函数。你可以对它取地址（&），它通常可以出现在赋值语句的左边。比如，变量、函数返回的左值引用等。

右值是一个表达式结束后就不复存在的临时值。你不能对它取地址（临时对象除外，但通常无意义），它通常只能出现在赋值语句的右边。比如，字面量（10, true）、函数返回的非引用类型值、a + b 的结果等。

在c++11之前，我们只有左值引用（&），它只能绑定到左值。尝试将一个左值引用绑定到右值（如临时对象）会导致编译错误（除非是 const 左值引用）。

```c++
int& ref = 10; // 编译错误！不能将非const左值引用绑定到右值
const int& const_ref = 10; // 正确。const左值引用可以延长临时对象的生命周期
```

c++11之后引入了右值引用(&&),它的核心目的就是识别出右值，并能够绑定到右值，从而允许我们对临时对象进行特殊操作。例如：

```c++
std::string s1 = "hello";
// std::string& l_ref = s1 + " world"; // 编译错误：s1 + " world" 是右值
std::string&& r_ref = s1 + " world"; // 正确：右值引用绑定到临时对象
r_ref += "!"; // 正确：我们可以修改这个临时对象
std::cout << r_ref << std::endl; // 输出 "hello world!"
```

这里要注意一个非常容易混淆的点。

> ##### TIP
>
> 如果一个右值引用本身有名字，那么这个引用变量本身是一个左值。
{: .block-tip }

例如上面例子中的r_ref本身就是一个左值，因为 r_ref 作为一个变量，它在表达式结束后依然存在，你可以多次访问它，所以它符合左值的定义。这个特性是理解 std::move 和完美转发的基础。

具体区别参考：

```
https://www.cnblogs.com/gqtcgq/p/9828247.html

https://www.internalpointers.com/post/understanding-meaning-lvalues-and-rvalues-c

https://cppreference.com/w/cpp/language/value_category.html
```

## 二、 std::move

移动语义要解决的问题是深拷贝带来的性能损耗，也就是说std::move是为了避免不必要的深拷贝而设计的。
例如：假设存在一个字符串的类: MyString，在 vector中push_back(MyString("world")) 这个场景中，我们创建了一个临时 MyString 对象，然后 vector 为了将它存入内部，不得不执行一次深拷贝。这个临时对象在 push_back 执行完毕后立刻被销毁。这里的拷贝是巨大的浪费：我们分配了新内存，逐字节复制了数据，而那个源数据（来自临时对象）马上就被丢弃了。如果字符串非常长，这个开销会非常大。

移动语义的核心思想是：对于一个即将被销毁的对象（右值），我们不拷贝它的资源，而是“窃取”或“移动”它的资源。

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

所以看出std::move只是对参数类型做了一个强制转换，转换为右值引用类型，std::move并不能移动任何东西，它唯一的功能是将一个左值或者右值强制转化为右值引用，继而可以通过右值引用使用该值，以用于移动语义。

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

例如：

```c++
MyString s1("hello");
// MyString s2 = s1; // 调用拷贝构造，因为 s1 是左值

MyString s3 = std::move(s1); // 1. std::move(s1) 将左值 s1 转换为右值引用
                           // 2. MyString 的构造函数重载解析选择了移动构造函数
                           // 3. s1 的资源被“移动”到 s3

// 警告！此时 s1 的内部指针已为 nullptr。
// s1 处于一个“有效但未指定”的状态。
// 对它的任何操作（除了析构或重新赋值）都可能是危险的。
```

参考

```
 https://blog.csdn.net/p942005405/article/details/84644069

《Effective.Modern.C++-42.Specific.Ways.to.Improve.Your.Use.of.C++11.and.C++14-2014》 Item23
```

## 三、 std::forward

完美转发是右值引用在泛型编程中的顶级应用。
例如下面的包装函数:

```c++
void target(MyString& s) { std::cout << "target(lvalue ref)" << std::endl; }
void target(MyString&& s) { std::cout << "target(rvalue ref)" << std::endl; }

template<typename T>
void wrapper(T param) {
    // 问题：这里的 param 是左值还是右值？
    // 答：param 是一个有名字的变量，它永远是左值！
    target(param); // 所以这里永远调用 target(lvalue ref)
}

int main() {
    MyString s;
    wrapper(s); // 期望调用 target(lvalue ref) -> 正确
    wrapper(MyString("temp")); // 期望调用 target(rvalue ref) -> 错误！实际调用了 lvalue 版本
}
```

问题就在于，无论 wrapper 的实参是左值还是右值，其形参 param 在 wrapper 函数体内都是一个有名字的变量，因此它是一个左值。这就导致了值类型的“退化”，我们丢失了原始参数是左值还是右值的信息。

> ##### TIP
>
> C++11规定了一种特殊情况，称为转发引用（也叫通用引用，Universal Reference）：
> 当一个函数模板的参数类型是 T&&，其中 T 是需要被推导的模板参数时，这个参数就是转发引用。
{: .block-tip }

转发引用能同时接受左值和右值，并在类型 T 中进行上述的引用折叠。
解决上述例子的问题的方法如下：

```c++
template<typename T>
void wrapper(T&& param) { // param 是一个转发引用
    // std::forward<T> 会检查 T 的类型：
    // 如果 T 是左值引用类型（U&），它返回一个左值引用
    // 如果 T 是非引用类型（U），它返回一个右值引用
    target(std::forward<T>(param));
}

int main() {
    MyString s;
    wrapper(s); // T 推导为 MyString&, forward返回左值引用 -> 调用 target(lvalue ref)
    wrapper(MyString("temp")); // T 推导为 MyString, forward返回右值引用 -> 调用 target(rvalue ref)
}
```

完美转发（Perfect Forwarding）就是利用转发引用和 std::forward 实现的，它在C++标准库中被广泛应用，例如 std::make_unique, std::vector::emplace_back 等，实现了参数传递的零成本转发。

std::forward大致原型为：

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
