# Blog Taxonomy Design

## Goal

Reorganize all 24 technical, mathematics, AI, and reading entries into a consistent taxonomy without changing article bodies or public article URLs. Every category and tag must produce a valid archive link, appear in the site search index, and remain enforceable for future posts.

## Content model

- Keep the four existing collections as the primary sections: `posts` (Programming), `AI`, `math`, and `reading`.
- Give every entry exactly one collection-scoped secondary category.
- Give every entry two to four normalized tags chosen from a controlled vocabulary.
- Do not use a tag to repeat the collection name or to record provenance.
- Record provenance with `source_type`. Omit it for original writing; use `source_type: repost` for translated, reproduced, or substantially adapted writing.
- Keep `源码解读` as a searchable content-form tag because it is useful for discovering code-reading articles.
- Preserve every filename, collection, layout, date, title, description, article body, and generated article URL.
- Old category and tag archive URLs do not require redirects. Jekyll will regenerate archives from the new front matter.

## Controlled taxonomy

The canonical vocabulary will live in `_data/taxonomy.yml`. Categories and tags are scoped per collection so a term is valid only where it is meaningful.

### AI

| Article | Category | Tags |
| --- | --- | --- |
| ByteTransformer源码解读 | GPU 与高性能计算 | BERT, CUDA, 源码解读 |
| 关于SM Occupancy的一些解释说明 | GPU 与高性能计算 | CUDA, GPU, 性能分析 |
| Attention系列整理Part1-基础 | 大模型基础 | Attention, Transformer, 模型原理 |
| Attention系列整理Part2-FlashAttention | GPU 与高性能计算 | Attention, FlashAttention, CUDA, GPU |
| 旋转位置编码 | 大模型基础 | RoPE, Transformer, 位置编码 |
| 生成式扩散模型DDPM及其推导 | 生成模型 | Diffusion, DDPM, 模型原理 |
| 生成式扩散模型采样方法DDIM及其推导 | 生成模型 | Diffusion, DDIM, 采样算法 |
| 对CUTE中的Thread-Value Layout的理解 | GPU 与高性能计算 | CuTe, CUTLASS, CUDA, 张量布局 |
| 解析FlashAttention2源码 | GPU 与高性能计算 | FlashAttention, CuTe, CUDA, 源码解读 |
| AIGC 模型量化工具调研报告 | 模型优化 | 模型量化, LLM, Diffusion, 推理优化 |
| vLLM V1 源码阅读笔记 | 大模型系统 | vLLM, LLM, 推理系统, 源码解读 |

### Programming

| Article | Category | Tags | Source type |
| --- | --- | --- | --- |
| c++中的 move 和 forward | C++ 语言 | C++, 现代C++, 移动语义 | original |
| c++中的 mutable关键字 | C++ 语言 | C++, 现代C++, 语言特性 | original |
| PyZMQ的简单入门 | 分布式与并发 | Python, ZeroMQ, 网络通信, 异步编程 | original |
| Windows中Linux常用命令的替代方案 | 开发工具 | Windows, PowerShell, 命令行 | repost |
| python中多进程实现任务调度 | 分布式与并发 | Python, 多进程, 任务调度, 系统设计 | original |
| 揭开浮点数的面纱 | 计算机基础 | 浮点数, IEEE 754, 数值计算 | repost |
| 揭开CUDA Stream的面纱 | GPU 与高性能计算 | CUDA, GPU, 并发, 性能优化 | original |

### Math

| Article | Category | Tags | Source type |
| --- | --- | --- | --- |
| 如何通过4个2构造任意整数 | 趣味数学 | 数论, 数学构造 | repost |
| 几个不等式的几何证明 | 数学可视化 | 不等式, 几何证明, AM-GM | repost |

### Reading

| Article | Category | Tags | Source type |
| --- | --- | --- | --- |
| 卡尔维诺：不确定世界中的旅行者 | 文学评论 | 文学, 卡尔维诺, 文学史 | repost |
| 看不见的城市 | 小说 | 文学, 卡尔维诺, 后现代主义 | original |
| 帕洛马尔 | 小说 | 文学, 卡尔维诺, 哲思小说 | original |
| The Godfather | 小说 | 文学, 马里奥·普佐, 犯罪小说 | original |

## Display and archive behavior

- Replace the manually maintained tag arrays in `_config.yml` with `_data/taxonomy.yml` as the single vocabulary source.
- A reusable collection taxonomy include will render categories and tags for a section. It will show only terms used by at least one entry, so every displayed link points to a generated archive page.
- Programming, AI, Math, and Reading landing pages will use the same include and collection-scoped archive URL convention already used by `jekyll-archives-v2`.
- Existing post and book layouts will continue to render each entry's category and tag links. No archive permalink format changes are required.

## Search behavior

- Extend each article's Ninja Keys search action with its category, tags, and source label as search keywords.
- Keep the visible title and description unchanged. Taxonomy terms are search metadata rather than text appended to the article description.
- Verify the vendored Ninja Keys data contract before choosing the keyword field. If version 1.2.11 does not search a dedicated keyword property, add a small local search adapter that includes taxonomy terms while keeping them visually hidden.
- The generated `assets/js/search-data.js` must contain all assigned categories and tags for the corresponding entries.

## Validation and failure handling

A repository-local validator will parse `_data/taxonomy.yml` and the front matter of all four content collections. It will fail with a file-specific message when:

- an entry does not have exactly one category;
- an entry has fewer than two or more than four tags;
- a category or tag is outside the collection's controlled vocabulary;
- tags are duplicated;
- `转载` remains as a tag;
- `source_type` is present with a value other than `repost`;
- taxonomy configuration contains a category or tag that is unused;
- a generated archive page is missing for any assigned category or tag;
- generated search data omits an entry's category or tags.

The metadata checks run before Jekyll build and do not require generated files. Archive and search-output checks run after build. GitHub Actions will run both layers, and the same commands will be available locally.

## Testing and acceptance criteria

1. A deliberately invalid fixture proves the validator catches cardinality, vocabulary, provenance, and unused-term errors before implementation is accepted.
2. All 24 entries pass the metadata validator after migration.
3. `bundle exec jekyll build` completes successfully.
4. Every assigned category and tag has a generated HTML archive page under its collection path.
5. Every article's generated search action contains its title plus category and tag terms.
6. Prettier, existing Python tests, and `git diff --check` pass.
7. A diff of article files shows front matter-only changes; bodies are byte-for-byte unchanged.

## Out of scope

- Redirects for old category or tag archives.
- Rewriting article prose, titles, dates, descriptions, or filenames.
- Moving entries between the four collections.
- Replacing Jekyll Archives or Ninja Keys.
