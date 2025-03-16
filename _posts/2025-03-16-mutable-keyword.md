---
layout: post
title: c++中的 mutable关键字
date: 2025-03-15 17:10:00
description: 简单解释一下c++中的 mutable关键字。
tags: c++
categories: programming
giscus_comments: false
related_posts: false
---


mutable 是用来修饰一个 const 示例的部分可变的数据成员的。如果要说得更清晰一点，就是说 mutable 的出现，将 C++ 中的 const 的概念{% sidenote 'One' 'See [cppreference cv](https://en.cppreference.com/w/cpp/language/cv)' %} 分成了两种：

- 二进制层面的 const，也就是「绝对的」常量，在任何情况下都不可修改(除非用 const_cast)

- 引入 mutable 之后，C++ 可以有逻辑层面的 const，也就是对一个常量实例来说，从外部观察，它是常量而不可修改；但是内部可以有非常量的状态。当然，所谓的「逻辑 const」，在 C++ 标准中并没有这一称呼。这只是为了方便理解，而创造出来的名词。

显而易见，mutable 只能用来修饰类的数据成员；而被 mutable 修饰的数据成员，可以在 const 成员函数中修改。

比如：

```c++
class HashTable {
 public:
    //...
    std::string lookup(const std::string& key) const
    {
        if (key == last_key_) {
            return last_value_;
        }

        std::string value{this->lookupInternal(key)};

        last_key_   = key;
        last_value_ = value;

        return value;
    }

 private:
    mutable std::string last_key_
    mutable std::string last_value_;
};
```

lookup 函数是const 的函数，正常来说是不能修改成员变量的，但是加上mutable之后，就可以修改了。

更实际的例子是用在锁上面。比如：

```c++
class Queue {
 public:
    void push(int x);
    bool empty() const;
};

```
假设这是一个线程安全的队列，empty函数是const的函数 不能修改内部成员，如果有锁，也就不能修改锁的状态，这样也就无法实现线程安全了。mutable可以实现这种情况。

```c++
class Queue {
private:
    mutable std::mutex mutex_;
 public:
    void push(int x);
    bool empty() const {
        std::lock_guard<std::mutex> lk(mutex_);
    }
};

```

还有一种情况使用mutable， 就是lambda函数：

在 Lambda 表达式的设计中，捕获变量有几种方式；其中按值捕获（Caputre by Value）的方式不允许程序员在 Lambda 函数的函数体中修改捕获的变量。而以 mutable 修饰 Lambda 函数，则可以打破这种限制。

```c++
int x{0};
auto f1 = [=]() mutable {x = 42;};  // okay, 创建了一个函数类型的实例
auto f2 = [=]()         {x = 42;};  // error, 不允许修改按值捕获的外部变量的值
```

需要注意的是，上述 f1 的函数体中，虽然我们给 x 做了赋值操作，但是这一操作仅只在函数内部生效——即，实际是给拷贝至函数内部的 x 进行赋值——而外部的 x 的值依旧是 0。
