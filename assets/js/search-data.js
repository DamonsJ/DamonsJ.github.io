// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-math",
          title: "Math",
          description: "瓦尔德拉达是一对孪生城市，一座在湖畔，另外一座在湖中如同倒影，湖畔的城市的每一个细节都会在水中那个城市完整的再现出来。这对孪生的城市并不相同，因为瓦尔德拉达出现或发生的一切都是不对称的；每个面孔和姿态，在镜子里都有呼应的面孔和姿态，可是它们是颠倒了的。两个瓦尔德拉达互相依存，目光相接，却并不相爱。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/math/";
          },
        },{id: "nav-programming",
          title: "Programming",
          description: "城市就像一块海绵，吸汲着这些不断涌流的记忆的潮水，并且随之膨胀着。对今日扎伊拉的描述，还应该包含扎伊拉的整个过去。然而，城市不会泄露自己的过去，只会把它像手纹一样藏起来，它被写在街巷的角落、窗格的护栏、楼梯的扶手、避雷的天线和旗杆上，每一道印记都是抓挠、锯锉、刻凿、猛击留下的痕迹",
          section: "Navigation",
          handler: () => {
            window.location.href = "/programming/";
          },
        },{id: "nav-ai",
          title: "AI",
          description: "关于宝琪的居民，有三种假设：他们憎恨地球；他们敬畏地球，乃至尽量避免与地面的任何接触；他们喜欢自己出生之前的地球，以至利用各种望远镜不知疲倦地观察着每一片树叶，每一块石子，每一只蚂蚁，着迷地冥思自己杳然的存在。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/AI/";
          },
        },{id: "nav-reading",
          title: "Reading",
          description: "佐贝伊德，月光下的白色城市，是梦境中的城市。其他国家的人们因为梦来到这里，改变这座城市，使她更接近梦境。最早来的人们想不通，是什么吸引那些人来佐贝伊德，走进这个陷阱，这座丑陋的城市。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/reading/";
          },
        },{id: "post-cuda-stream和并发",
      
        title: "CUDA Stream和并发",
      
      taxonomySource: "_posts/2026-05-08-cuda-streams-and-concurrency.md",
      keywords: "GPU 与高性能计算 CUDA GPU 并发 性能优化",
      description: "从基础概念到 CUDA Graphs / Stream-Ordered Allocator / Hopper 异步特性，系统梳理 CUDA Stream 的过去、现在与今天的最佳实践。 GPU 与高性能计算 CUDA GPU 并发 性能优化",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2026/05/08/cuda-streams-and-concurrency.html";
        
      },
    },{id: "post-揭开浮点数的面纱",
      
        title: "揭开浮点数的面纱",
      
      taxonomySource: "_posts/2026-05-06-floating-point.md",
      keywords: "计算机基础 浮点数 IEEE 754 数值计算 转载",
      description: "转载一篇浮点数的记录。 计算机基础 浮点数 IEEE 754 数值计算 转载",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2026/05/07/floating-point.html";
        
      },
    },{id: "post-python中多进程实现任务调度",
      
        title: "python中多进程实现任务调度",
      
      taxonomySource: "_posts/2026-04-05-python-task-scheduler.md",
      keywords: "分布式与并发 Python 多进程 任务调度 系统设计",
      description: "记录python中实现多进程任务调度的工程实现,其中的进程调度架构类似vllm中的实现。 分布式与并发 Python 多进程 任务调度 系统设计",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2026/04/06/python-task-scheduler.html";
        
      },
    },{id: "post-最常用的-linux-命令的-windows-对应命令",
      
        title: "最常用的 Linux 命令的 Windows 对应命令",
      
      taxonomySource: "_posts/2026-04-02-linux-commands-in-windows.md",
      keywords: "开发工具 Windows PowerShell 命令行 转载",
      description: "记录最常用的 Linux 命令的 Windows 对应命令。 开发工具 Windows PowerShell 命令行 转载",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2026/04/03/linux-commands-in-windows.html";
        
      },
    },{id: "post-pyzmq的简介",
      
        title: "PyZMQ的简介",
      
      taxonomySource: "_posts/2026-01-17-pyzmq-introduction.md",
      keywords: "分布式与并发 Python ZeroMQ 网络通信 异步编程",
      description: "使用PyZMQ过程中的记录。 分布式与并发 Python ZeroMQ 网络通信 异步编程",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2026/01/18/pyzmq-introduction.html";
        
      },
    },{id: "post-c-中的-mutable关键字",
      
        title: "c++中的 mutable关键字",
      
      taxonomySource: "_posts/2025-03-16-mutable-keyword.md",
      keywords: "C++ 语言 C++ 现代C++ 语言特性",
      description: "简单解释一下c++中的 mutable关键字。 C++ 语言 C++ 现代C++ 语言特性",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2025/03/16/mutable-keyword.html";
        
      },
    },{id: "post-c-中的-move-和-forward",
      
        title: "c++中的 move 和 forward",
      
      taxonomySource: "_posts/2025-03-09-move-forward.md",
      keywords: "C++ 语言 C++ 现代C++ 移动语义",
      description: "介绍c++中的move和forward的区别及各自的实现方式。 C++ 语言 C++ 现代C++ 移动语义",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2025/03/09/move-forward.html";
        
      },
    },{id: "AI-bytetransformer源码解读",
          title: 'ByteTransformer源码解读',
          taxonomySource: "_AI/2023-09-14-bytetransformer.md",
          keywords: "GPU 与高性能计算 BERT CUDA 源码解读",
          description: "记录学习ByteTransformer过程中的代码释疑 GPU 与高性能计算 BERT CUDA 源码解读",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2023-09-14-bytetransformer.html";
            },},{id: "AI-关于sm-occupancy的一些解释说明",
          title: '关于SM Occupancy的一些解释说明',
          taxonomySource: "_AI/2025-03-22-something-about-sm-occupancy.md",
          keywords: "GPU 与高性能计算 CUDA GPU 性能分析",
          description: "解释SM Occupancy的计算和意义 GPU 与高性能计算 CUDA GPU 性能分析",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2025-03-22-something-about-sm-occupancy.html";
            },},{id: "AI-attention系列整理part1-基础",
          title: 'Attention系列整理Part1-基础',
          taxonomySource: "_AI/2025-04-02-rethinking-attention-1.md",
          keywords: "大模型基础 Attention Transformer 模型原理",
          description: "整理Attention的基础知识，推导计算量 大模型基础 Attention Transformer 模型原理",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2025-04-02-rethinking-attention-1.html";
            },},{id: "AI-attention系列整理part2-flashattention",
          title: 'Attention系列整理Part2-FlashAttention',
          taxonomySource: "_AI/2025-04-26-rethinking-attention-2.md",
          keywords: "GPU 与高性能计算 Attention FlashAttention CUDA GPU",
          description: "整理FlashAttention系列以及其计算原理 GPU 与高性能计算 Attention FlashAttention CUDA GPU",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2025-04-26-rethinking-attention-2.html";
            },},{id: "AI-旋转位置编码",
          title: '旋转位置编码',
          taxonomySource: "_AI/2025-05-30-rope.md",
          keywords: "大模型基础 RoPE Transformer 位置编码",
          description: "整理旋转位置编码相关的内容，介绍其来源及具体的算法 大模型基础 RoPE Transformer 位置编码",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2025-05-30-rope.html";
            },},{id: "AI-生成式扩散模型ddpm及其推导",
          title: '生成式扩散模型DDPM及其推导',
          taxonomySource: "_AI/2025-06-18-DDPM.md",
          keywords: "生成模型 Diffusion DDPM 模型原理",
          description: "把学习扩散模型DDPM的过程记录一下 生成模型 Diffusion DDPM 模型原理",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2025-06-18-DDPM.html";
            },},{id: "AI-生成式扩散模型采样方法ddim及其推导",
          title: '生成式扩散模型采样方法DDIM及其推导',
          taxonomySource: "_AI/2025-07-16-DDIM.md",
          keywords: "生成模型 Diffusion DDIM 采样算法",
          description: "记录学习DDIM采样方法的过程 生成模型 Diffusion DDIM 采样算法",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2025-07-16-DDIM.html";
            },},{id: "AI-对cute中的thread-value-layout的理解",
          title: '对CUTE中的Thread-Value Layout的理解',
          taxonomySource: "_AI/2025-12-10-CUTE-Thread-Value-Layout.md",
          keywords: "GPU 与高性能计算 CuTe CUTLASS CUDA 张量布局",
          description: "记录一下学习cute过程中的对Thread-Value Layout的理解 GPU 与高性能计算 CuTe CUTLASS CUDA 张量布局",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2025-12-10-CUTE-Thread-Value-Layout.html";
            },},{id: "AI-解析flashattention2源码",
          title: '解析FlashAttention2源码',
          taxonomySource: "_AI/2026-01-05-Understanding-FlashAttention2-SourceCode.md",
          keywords: "GPU 与高性能计算 FlashAttention CuTe CUDA 源码解读",
          description: "记录一下学习FlashAttention2源码过程中对一些概念的理解 GPU 与高性能计算 FlashAttention CuTe CUDA 源码解读",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2026-01-05-Understanding-FlashAttention2-SourceCode.html";
            },},{id: "AI-aigc-模型量化工具调研报告",
          title: 'AIGC 模型量化工具调研报告',
          taxonomySource: "_AI/2026-04-15-quantization_tools_research.md",
          keywords: "模型优化 模型量化 LLM Diffusion 推理优化",
          description: "对主流量化工具的调研与对比 模型优化 模型量化 LLM Diffusion 推理优化",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2026-04-15-quantization_tools_research.html";
            },},{id: "AI-vllm-v1-源码阅读笔记",
          title: 'vLLM V1 源码阅读笔记',
          taxonomySource: "_AI/2026-04-16-reading_about_vllm.md",
          keywords: "大模型系统 vLLM LLM 推理系统 源码解读",
          description: "阅读 vLLM V1 源码后的架构理解与核心流程记录 大模型系统 vLLM LLM 推理系统 源码解读",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2026-04-16-reading_about_vllm.html";
            },},{id: "AI-aigc-分布式推理-从-wan2-2-理解-dp-tp-cp-sp-pp",
          title: 'AIGC 分布式推理：从 Wan2.2 理解 DP/TP/CP/SP/PP',
          taxonomySource: "_AI/2026-08-08-distribute-aigc.md",
          keywords: "生成模型 分布式推理 Diffusion 视频生成 推理优化",
          description: "以 Wan2.2 TI2V 5B 为例，记录分布式推理中各种并行方式切什么、怎么通信、代价有多大 生成模型 分布式推理 Diffusion 视频生成 推理优化",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2026-08-08-distribute-aigc.html";
            },},{id: "math-如何通过4个2构造任意整数",
          title: '如何通过4个2构造任意整数',
          taxonomySource: "_math/2025-03-08-making-any-integer-with-four-2s.md",
          keywords: "趣味数学 数论 数学构造 转载",
          description: "介绍一种方法，通过4个2构造任意整数 趣味数学 数论 数学构造 转载",
          section: "Math",handler: () => {
              window.location.href = "/math/2025-03-08-making-any-integer-with-four-2s.html";
            },},{id: "math-几个不等式的几何证明",
          title: '几个不等式的几何证明',
          taxonomySource: "_math/2026-03-21-The-Shape-of-Inequalities.md",
          keywords: "数学可视化 不等式 几何证明 AM-GM 转载",
          description: "介绍几个不等式的可视化几何证明方法 数学可视化 不等式 几何证明 AM-GM 转载",
          section: "Math",handler: () => {
              window.location.href = "/math/2026-03-21-The-Shape-of-Inequalities.html";
            },},{id: "math-快速平方根倒数-0x5f3759df-这个魔法数是怎么来的",
          title: '快速平方根倒数：0x5f3759df 这个魔法数是怎么来的',
          taxonomySource: "_math/2026-08-13-fast-inverse-square-root.md",
          keywords: "数值计算 平方根倒数 IEEE 754 牛顿迭代 数值计算",
          description: "从 IEEE 754 的位表示出发，通俗推导快速平方根倒数中的魔法数与牛顿迭代 数值计算 平方根倒数 IEEE 754 牛顿迭代 数值计算",
          section: "Math",handler: () => {
              window.location.href = "/math/2026-08-13-fast-inverse-square-root.html";
            },},{id: "projects-project-1",
          title: 'project 1',
          taxonomySource: "_projects/1_project.md",
          keywords: "work ",
          description: "with background image work ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project.html";
            },},{id: "projects-project-2",
          title: 'project 2',
          taxonomySource: "_projects/2_project.md",
          keywords: "work ",
          description: "a project with a background image and giscus comments work ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project.html";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          taxonomySource: "_projects/3_project.md",
          keywords: "work ",
          description: "a project that redirects to another website work ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project.html";
            },},{id: "projects-project-4",
          title: 'project 4',
          taxonomySource: "_projects/4_project.md",
          keywords: "fun ",
          description: "another without an image fun ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project.html";
            },},{id: "projects-project-5",
          title: 'project 5',
          taxonomySource: "_projects/5_project.md",
          keywords: "fun ",
          description: "a project with a background image fun ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project.html";
            },},{id: "projects-project-6",
          title: 'project 6',
          taxonomySource: "_projects/6_project.md",
          keywords: "fun ",
          description: "a project with no image fun ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project.html";
            },},{id: "projects-project-7",
          title: 'project 7',
          taxonomySource: "_projects/7_project.md",
          keywords: "work ",
          description: "with background image work ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project.html";
            },},{id: "projects-project-8",
          title: 'project 8',
          taxonomySource: "_projects/8_project.md",
          keywords: "work ",
          description: "an other project with a background image and giscus comments work ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project.html";
            },},{id: "projects-project-9",
          title: 'project 9',
          taxonomySource: "_projects/9_project.md",
          keywords: "fun ",
          description: "another project with an image 🎉 fun ",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project.html";
            },},{id: "reading-看不见的城市",
          title: '看不见的城市',
          taxonomySource: "_reading/invisible-cites.md",
          keywords: "小说 文学 卡尔维诺 后现代主义",
          description: " 小说 文学 卡尔维诺 后现代主义",
          section: "Reading",handler: () => {
              window.location.href = "/reading/invisible-cites.html";
            },},{id: "reading-帕洛马尔",
          title: '帕洛马尔',
          taxonomySource: "_reading/palomar.md",
          keywords: "小说 文学 卡尔维诺 哲思小说",
          description: " 小说 文学 卡尔维诺 哲思小说",
          section: "Reading",handler: () => {
              window.location.href = "/reading/palomar.html";
            },},{id: "reading-卡尔维诺-不确定世界中的旅行者",
          title: '卡尔维诺：不确定世界中的旅行者',
          taxonomySource: "_reading/calvino-traveller-uncertainty.md",
          keywords: "文学评论 文学 卡尔维诺 文学史 转载",
          description: " 文学评论 文学 卡尔维诺 文学史 转载",
          section: "Reading",handler: () => {
              window.location.href = "/reading/calvino-traveller-uncertainty.html";
            },},{id: "reading-读-通向蜘蛛巢的小径-之前-卡尔维诺-抵抗运动与一个孩子眼中的战争",
          title: '读《通向蜘蛛巢的小径》之前：卡尔维诺、抵抗运动与一个孩子眼中的战争',
          taxonomySource: "_reading/before-reading-the-path-to-the-spiders-nests.md",
          keywords: "小说 文学 卡尔维诺",
          description: " 小说 文学 卡尔维诺",
          section: "Reading",handler: () => {
              window.location.href = "/reading/before-reading-the-path-to-the-spiders-nests.html";
            },},{id: "reading-通向蜘蛛巢的小径",
          title: '通向蜘蛛巢的小径',
          taxonomySource: "_reading/the-path-to-the-spiders-nests.md",
          keywords: "小说 文学 卡尔维诺",
          description: " 小说 文学 卡尔维诺",
          section: "Reading",handler: () => {
              window.location.href = "/reading/the-path-to-the-spiders-nests.html";
            },},{id: "reading-读-烟云-阿根廷蚂蚁-之前-两种无法清除的现代生活",
          title: '读《烟云·阿根廷蚂蚁》之前：两种无法清除的现代生活',
          taxonomySource: "_reading/before-reading-smog-and-the-argentine-ant.md",
          keywords: "小说 文学 卡尔维诺",
          description: " 小说 文学 卡尔维诺",
          section: "Reading",handler: () => {
              window.location.href = "/reading/before-reading-smog-and-the-argentine-ant.html";
            },},{id: "reading-烟云-阿根廷蚂蚁-生活在无法清除的困境里",
          title: '《烟云·阿根廷蚂蚁》：生活在无法清除的困境里',
          taxonomySource: "_reading/smog-and-the-argentine-ant.md",
          keywords: "小说 文学 卡尔维诺",
          description: " 小说 文学 卡尔维诺",
          section: "Reading",handler: () => {
              window.location.href = "/reading/smog-and-the-argentine-ant.html";
            },},{id: "reading-读-分成两半的子爵-之前-幻想如何重新进入现实",
          title: '读《分成两半的子爵》之前：幻想如何重新进入现实',
          taxonomySource: "_reading/before-reading-the-cloven-viscount.md",
          keywords: "小说 文学 卡尔维诺 文学史",
          description: " 小说 文学 卡尔维诺 文学史",
          section: "Reading",handler: () => {
              window.location.href = "/reading/before-reading-the-cloven-viscount.html";
            },},{id: "reading-分成两半的子爵-完整不是没有受伤",
          title: '《分成两半的子爵》：完整不是没有受伤',
          taxonomySource: "_reading/Il-viscount-dimezzato.md",
          keywords: "小说 文学 卡尔维诺 文学史",
          description: " 小说 文学 卡尔维诺 文学史",
          section: "Reading",handler: () => {
              window.location.href = "/reading/Il-viscount-dimezzato.html";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%73%68%6C%6B%6C%39%39@%31%36%33.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/DamonsJ# your GitHub user name", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-weibo_username',
        title: 'Weibo_username',
        section: 'Socials',
        handler: () => {
          window.open("", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.similarity.cn/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
