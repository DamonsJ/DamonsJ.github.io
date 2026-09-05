# 《分成两半的子爵》读后感改写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留作者观点、第一人称语气和两段引文的前提下，把《分成两半的子爵》读后感改写成一篇简洁、论证连贯的博客文章。

**Architecture:** 只修改一篇 Markdown 文章。论证依次经过“战争造成分裂—善恶两半都片面—复原是经验的重新结合—完整不是没有受伤”，每段情节只证明一个判断。

**Tech Stack:** Jekyll、Markdown、Ruby 内容校验脚本

---

### Task 1: 重写文章正文

**Files:**
- Modify: `_reading/Il-viscount-dimezzato.md`

- [ ] **Step 1: 调整标题并保留元数据**

把标题改为 `《分成两半的子爵》：完整不是没有受伤`；保留封面、ISBN、评分、豆瓣条目和阅读状态。

- [ ] **Step 2: 重排开头**

保留作者读完三本卡尔维诺作品后的感受和“为什么要把子爵分成两半”的疑问，删除“新现实主义—寓言—幻想”的线性分期，把问题直接落到“分裂为何比完整更能让人认识自己”。

- [ ] **Step 3: 建立四段论证**

按以下顺序写出正文：

1. 炮弹先把一个人从身体上劈开，表明分裂首先来自战争和外部力量，而非人物主动选择善恶。
2. 恶的一半把自己看到的事物也劈开，使自身残缺成为其他人的生活方式；善的一半则把绝对善意变成持续的道德要求。二者都把具体的人压缩为单一原则。
3. 决斗和复原不是善消灭恶，而是两个片面部分重新结合；复原后的子爵因为保留两半的经历而变得更明智，并非回到受伤前的天真完整。
4. 以“仅仅一个完整的子爵不足以使全世界变得完整”收束：社会仍会伤害人，完整意味着不让伤害或角色永久规定自己只能成为哪一半。

- [ ] **Step 4: 保留并安放两段引文**

完整保留草稿中的两段引文。第一段放在提出“分裂带来认识”之后，第二段放在解释“复原并非回到原样”之后；引文使用 Markdown 引用块，前后各有明确分析。

- [ ] **Step 5: 保持作者语气并控制篇幅**

保留“我一开始……后来才……”和直接向读者发问的表达。正文含引文控制在约 1800—2500 个汉字，不新增大段历史背景，也不逐项复述情节。

### Task 2: 内容自检

**Files:**
- Check: `_reading/Il-viscount-dimezzato.md`

- [ ] **Step 1: 检查核心要求**

运行：

```bash
ruby -e 's=File.read("_reading/Il-viscount-dimezzato.md"); b=s.sub(/\A---.*?---\s*/m, ""); puts b.scan(/[\u4e00-\u9fff]/).length; puts b.scan(/^## /).length; puts b.scan(/^> /).length'
```

预期：汉字数为 1800—2500；文章有清楚的小标题；两段引文均被保留为引用块。

- [ ] **Step 2: 检查格式差异**

运行：

```bash
git diff --check
```

预期：无输出，退出码为 0。

### Task 3: 验证博客构建

**Files:**
- Check: `_reading/Il-viscount-dimezzato.md`
- Check: `_site/reading/Il-viscount-dimezzato.html`

- [ ] **Step 1: 运行内容校验**

运行：

```bash
bundle exec ruby scripts/validate_taxonomy.rb
bundle exec ruby scripts/validate_medium_zoom.rb
```

预期：两个脚本均退出为 0。

- [ ] **Step 2: 构建站点**

运行：

```bash
bundle exec jekyll build --trace
```

预期：退出码为 0，并生成对应读书文章页面。

### Task 4: 提供针对性写作建议

- [ ] **Step 1: 根据修改前后的差异总结建议**

最终说明本次如何保留作者声音，并提供三到五条可复用的方法：每篇文章只回答一个问题、先写判断再选情节、让引文承担论证任务、区分“情节发生了什么”和“它说明什么”、结尾修正开头提出的观点。
