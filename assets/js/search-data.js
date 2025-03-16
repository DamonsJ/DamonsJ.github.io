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
        },{id: "nav-vision",
          title: "Vision",
          description: "佐贝伊德，月光下的白色城市，是梦境中的城市。其他国家的人们因为梦来到这里，改变这座城市，使她更接近梦境。最早来的人们想不通，是什么吸引那些人来佐贝伊德，走进这个陷阱，这座丑陋的城市。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/vision/";
          },
        },{id: "nav-computer",
          title: "Computer",
          description: "我想：“到了生命的某一个时刻，在你认识的人之中，已去世的会比活着的多。这时你的心就会拒绝接受更多的面孔和更多的表情，你遇见的每一张新面子都是旧的容貌，它们各自寻得合适的面具。” 我想：“也许阿德尔玛是你垂死时抵达的城市，每个人都可以在这里跟故人重逢。这就标志着我也是死人。” 我又想：“这意味着彼世并不快乐。”",
          section: "Navigation",
          handler: () => {
            window.location.href = "/computer/";
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
          description: "”符号形成一种语言，但那不是你们自以为了解的语言。” 我明白了，我必须从引导我追寻事物直至此地的形象中解脱出来：只有那时，我才能理解伊帕奇亚的语言。当我的灵魂只需要音乐的营养与刺激时，我晓得应该到墓地去：音乐家们都躲在墓穴中，笛子的颤音和竖琴的和弦在坟头间彼此呼应。当然，总有一天，我在伊帕奇亚的唯一愿望将是起身离去。我知道，不该走向海港码头，而必须爬上城堡最高的尖塔，去等候一条路经那里的船只。但是能否有船驶过呢？没有一种语言是绝对不骗人的。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/reading/";
          },
        },{id: "nav-life",
          title: "Life",
          description: "在他盼望着城市时，心里就会想到所有这一切。因此，伊西多拉便是他梦中的城市，但只有一点不同。在梦中的城市里，他正值青春，而到达伊西多拉城时，他已年老。广场上有一堵墙，老人们倚坐在那里看着过往的年轻人；他和这些老人并坐在一起。当初的欲望已是记忆。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/life/";
          },
        },{id: "post-c-中的-mutable关键字",
      
        title: "c++中的 mutable关键字",
      
      description: "简单解释一下c++中的 mutable关键字。",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2025/03/15/mutable-keyword.html";
        
      },
    },{id: "post-tufte-style-jekyll-blog",
      
        title: "Tufte-style Jekyll blog",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2025/03/15/tufte-style-jekyll-blog.html";
        
      },
    },{id: "post-c-中的-move-和-forward",
      
        title: "c++中的 move 和 forward",
      
      description: "介绍c++中的move和forward的区别及各自的实现方式。",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/programming/2025/03/08/move-forward.html";
        
      },
    },{id: "AI-a-post-with-test1",
          title: 'a post with test1',
          description: "test1",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-01-26-geojson-map.html";
            },},{id: "AI-a-post-with-vega-lite",
          title: 'a post with vega lite',
          description: "this is what 4444",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-01-27-vega-lite.html";
            },},{id: "AI-a-post-with-advanced-image-components",
          title: 'a post with advanced image components',
          description: "test2",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-01-27-advanced-images.html";
            },},{id: "AI-a-post-with-code-diff",
          title: 'a post with code diff',
          description: "this is how you can display code diffs333",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-01-27-code-diff.html";
            },},{id: "AI-a-post-with-pseudo-code",
          title: 'a post with pseudo code',
          description: "5666code could look like",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-04-15-pseudocode.html";
            },},{id: "AI-a-post-that-can-be-cited",
          title: 'a post that can be cited',
          description: "9999this is what a post that can be cited looks like",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-04-28-post-citation.html";
            },},{id: "AI-a-post-with-typograms",
          title: 'a post with typograms',
          description: "this is what included typograms code could look like",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-04-29-typograms.html";
            },},{id: "AI-a-post-with-tabs",
          title: 'a post with tabs',
          description: "this is what included tabs in a post could look like",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-05-01-tabs.html";
            },},{id: "AI-222243433",
          title: '222243433',
          description: "2222222",
          section: "Ai",handler: () => {
              window.location.href = "/AI/2024-12-04-photo-gallery.html";
            },},{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather.html";
            },},{id: "computer-a-post-with-test1",
          title: 'a post with test1',
          description: "test1",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-01-26-geojson-map.html";
            },},{id: "computer-a-post-with-vega-lite",
          title: 'a post with vega lite',
          description: "this is what 4444",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-01-27-vega-lite.html";
            },},{id: "computer-a-post-with-advanced-image-components",
          title: 'a post with advanced image components',
          description: "test2",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-01-27-advanced-images.html";
            },},{id: "computer-a-post-with-code-diff",
          title: 'a post with code diff',
          description: "this is how you can display code diffs333",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-01-27-code-diff.html";
            },},{id: "computer-a-post-with-pseudo-code",
          title: 'a post with pseudo code',
          description: "5666code could look like",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-04-15-pseudocode.html";
            },},{id: "computer-a-post-that-can-be-cited",
          title: 'a post that can be cited',
          description: "9999this is what a post that can be cited looks like",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-04-28-post-citation.html";
            },},{id: "computer-a-post-with-typograms",
          title: 'a post with typograms',
          description: "this is what included typograms code could look like",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-04-29-typograms.html";
            },},{id: "computer-a-post-with-tabs",
          title: 'a post with tabs',
          description: "this is what included tabs in a post could look like",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-05-01-tabs.html";
            },},{id: "computer-222243433",
          title: '222243433',
          description: "2222222",
          section: "Computer",handler: () => {
              window.location.href = "/computer/2024-12-04-photo-gallery.html";
            },},{id: "math-如何通过4个2构造任意整数",
          title: '如何通过4个2构造任意整数',
          description: "介绍一种方法，通过4个2构造任意整数",
          section: "Math",handler: () => {
              window.location.href = "/math/2025-03-08-making-any-integer-with-four-2s.html";
            },},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project.html";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project.html";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project.html";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project.html";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project.html";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project.html";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project.html";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project.html";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project.html";
            },},{id: "reading-a-post-with-test1",
          title: 'a post with test1',
          description: "test1",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-01-26-geojson-map.html";
            },},{id: "reading-a-post-with-vega-lite",
          title: 'a post with vega lite',
          description: "this is what 4444",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-01-27-vega-lite.html";
            },},{id: "reading-a-post-with-advanced-image-components",
          title: 'a post with advanced image components',
          description: "test2",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-01-27-advanced-images.html";
            },},{id: "reading-a-post-with-code-diff",
          title: 'a post with code diff',
          description: "this is how you can display code diffs333",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-01-27-code-diff.html";
            },},{id: "reading-a-post-with-pseudo-code",
          title: 'a post with pseudo code',
          description: "5666code could look like",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-04-15-pseudocode.html";
            },},{id: "reading-a-post-that-can-be-cited",
          title: 'a post that can be cited',
          description: "9999this is what a post that can be cited looks like",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-04-28-post-citation.html";
            },},{id: "reading-a-post-with-typograms",
          title: 'a post with typograms',
          description: "this is what included typograms code could look like",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-04-29-typograms.html";
            },},{id: "reading-a-post-with-tabs",
          title: 'a post with tabs',
          description: "this is what included tabs in a post could look like",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-05-01-tabs.html";
            },},{id: "reading-222243433",
          title: '222243433',
          description: "2222222",
          section: "Reading",handler: () => {
              window.location.href = "/reading/2024-12-04-photo-gallery.html";
            },},{id: "vision-a-post-with-test1",
          title: 'a post with test1',
          description: "test1",
          section: "Vision",handler: () => {
              window.location.href = "/vision/2024-01-26-geojson-map.html";
            },},{id: "vision-a-post-with-advanced-image-components",
          title: 'a post with advanced image components',
          description: "test2",
          section: "Vision",handler: () => {
              window.location.href = "/vision/2024-01-27-advanced-images.html";
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
          window.open("https://www.aipodcast.cn/", "_blank");
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