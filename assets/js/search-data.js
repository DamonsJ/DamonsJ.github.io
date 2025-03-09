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
          description: "write something about programming",
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
          description: "Write something about Computer",
          section: "Navigation",
          handler: () => {
            window.location.href = "/computer/";
          },
        },{id: "nav-ai",
          title: "AI",
          description: "Write something about AI",
          section: "Navigation",
          handler: () => {
            window.location.href = "/AI/";
          },
        },{id: "nav-reading",
          title: "Reading",
          description: "Write something about Reading",
          section: "Navigation",
          handler: () => {
            window.location.href = "/reading/";
          },
        },{id: "nav-life",
          title: "Life",
          description: "Write something about life",
          section: "Navigation",
          handler: () => {
            window.location.href = "/life/";
          },
        },{id: "post-a-post-with-image-galleries",
      
        title: "a post with image galleries",
      
      description: "this is what included image galleries could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/12/04/photo-gallery.html";
        
      },
    },{id: "post-google-gemini-updates-flash-1-5-gemma-2-and-project-astra",
      
        title: 'Google Gemini updates: Flash 1.5, Gemma 2 and Project Astra <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "We’re sharing updates across our Gemini family of models and a glimpse of Project Astra, our vision for the future of AI assistants.",
      section: "Posts",
      handler: () => {
        
          window.open("https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/", "_blank");
        
      },
    },{id: "post-a-post-with-tabs",
      
        title: "a post with tabs",
      
      description: "this is what included tabs in a post could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/05/01/tabs.html";
        
      },
    },{id: "post-a-post-with-typograms",
      
        title: "a post with typograms",
      
      description: "this is what included typograms code could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/04/29/typograms.html";
        
      },
    },{id: "post-a-post-that-can-be-cited",
      
        title: "a post that can be cited",
      
      description: "this is what a post that can be cited looks like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/04/28/post-citation.html";
        
      },
    },{id: "post-a-post-with-pseudo-code",
      
        title: "a post with pseudo code",
      
      description: "this is what included pseudo code could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/04/15/pseudocode.html";
        
      },
    },{id: "post-a-post-with-code-diff",
      
        title: "a post with code diff",
      
      description: "this is how you can display code diffs",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/01/27/code-diff.html";
        
      },
    },{id: "post-a-post-with-advanced-image-components",
      
        title: "a post with advanced image components",
      
      description: "this is what advanced image components could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/01/27/advanced-images.html";
        
      },
    },{id: "post-a-post-with-vega-lite",
      
        title: "a post with vega lite",
      
      description: "this is what included vega lite code could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/01/27/vega-lite.html";
        
      },
    },{id: "post-a-post-with-geojson",
      
        title: "a post with geojson",
      
      description: "this is what included geojson code could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/01/26/geojson-map.html";
        
      },
    },{id: "post-a-post-with-echarts",
      
        title: "a post with echarts",
      
      description: "this is what included echarts code could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/01/26/echarts.html";
        
      },
    },{id: "post-a-post-with-chart-js",
      
        title: "a post with chart.js",
      
      description: "this is what included chart.js code could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2024/01/26/chartjs.html";
        
      },
    },{id: "post-a-post-with-tikzjax",
      
        title: "a post with TikZJax",
      
      description: "this is what included TikZ code could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/12/12/tikzjax.html";
        
      },
    },{id: "post-a-post-with-bibliography",
      
        title: "a post with bibliography",
      
      description: "an example of a blog post with bibliography",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/07/12/post-bibliography.html";
        
      },
    },{id: "post-a-post-with-jupyter-notebook",
      
        title: "a post with jupyter notebook",
      
      description: "an example of a blog post with jupyter notebook",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/07/04/jupyter-notebook.html";
        
      },
    },{id: "post-a-post-with-custom-blockquotes",
      
        title: "a post with custom blockquotes",
      
      description: "an example of a blog post with custom blockquotes",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/05/12/custom-blockquotes.html";
        
      },
    },{id: "post-a-post-with-table-of-contents-on-a-sidebar",
      
        title: "a post with table of contents on a sidebar",
      
      description: "an example of a blog post with table of contents on a sidebar",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/04/25/sidebar-table-of-contents.html";
        
      },
    },{id: "post-a-post-with-audios",
      
        title: "a post with audios",
      
      description: "this is what included audios could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/04/25/audios.html";
        
      },
    },{id: "post-a-post-with-videos",
      
        title: "a post with videos",
      
      description: "this is what included videos could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/04/24/videos.html";
        
      },
    },{id: "post-displaying-beautiful-tables-with-bootstrap-tables",
      
        title: "displaying beautiful tables with Bootstrap Tables",
      
      description: "an example of how to use Bootstrap Tables",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/03/20/tables.html";
        
      },
    },{id: "post-a-post-with-table-of-contents",
      
        title: "a post with table of contents",
      
      description: "an example of a blog post with table of contents",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2023/03/20/table-of-contents.html";
        
      },
    },{id: "post-a-post-with-giscus-comments",
      
        title: "a post with giscus comments",
      
      description: "an example of a blog post with giscus comments",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/external-services/2022/12/10/giscus-comments.html";
        
      },
    },{id: "post-displaying-external-posts-on-your-al-folio-blog",
      
        title: 'Displaying External Posts on Your al-folio Blog <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.open("https://medium.com/@al-folio/displaying-external-posts-on-your-al-folio-blog-b60a1d241a0a?source=rss-17feae71c3c4------2", "_blank");
        
      },
    },{id: "post-a-post-with-redirect",
      
        title: "a post with redirect",
      
      description: "you can also redirect to assets like pdf",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/assets/pdf/example_pdf.pdf";
        
      },
    },{id: "post-a-post-with-diagrams",
      
        title: "a post with diagrams",
      
      description: "an example of a blog post with diagrams",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/2021/07/04/diagrams.html";
        
      },
    },{id: "post-a-distill-style-blog-post",
      
        title: "a distill-style blog post",
      
      description: "an example of a distill-style blog post and main elements",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/2021/05/22/distill.html";
        
      },
    },{id: "post-a-post-with-twitter",
      
        title: "a post with twitter",
      
      description: "an example of a blog post with twitter",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/external-services/2020/09/28/twitter.html";
        
      },
    },{id: "post-a-post-with-disqus-comments",
      
        title: "a post with disqus comments",
      
      description: "an example of a blog post with disqus comments",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/external-services/2015/10/20/disqus-comments.html";
        
      },
    },{id: "post-a-post-with-math",
      
        title: "a post with math",
      
      description: "an example of a blog post with some math",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2015/10/20/math.html";
        
      },
    },{id: "post-a-post-with-code",
      
        title: "a post with code",
      
      description: "an example of a blog post with some code",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2015/07/15/code.html";
        
      },
    },{id: "post-a-post-with-images",
      
        title: "a post with images",
      
      description: "this is what included images could look like",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2015/05/15/images.html";
        
      },
    },{id: "post-a-post-with-formatting-and-links",
      
        title: "a post with formatting and links",
      
      description: "march &amp; april, looking forward to summer",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/sample-posts/2015/03/15/formatting-and-links.html";
        
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
            },},{id: "life-a-post-with-test1",
          title: 'a post with test1',
          description: "test1",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-01-26-geojson-map.html";
            },},{id: "life-a-post-with-vega-lite",
          title: 'a post with vega lite',
          description: "this is what 4444",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-01-27-vega-lite.html";
            },},{id: "life-a-post-with-advanced-image-components",
          title: 'a post with advanced image components',
          description: "test2",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-01-27-advanced-images.html";
            },},{id: "life-a-post-with-code-diff",
          title: 'a post with code diff',
          description: "this is how you can display code diffs333",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-01-27-code-diff.html";
            },},{id: "life-a-post-with-pseudo-code",
          title: 'a post with pseudo code',
          description: "5666code could look like",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-04-15-pseudocode.html";
            },},{id: "life-a-post-that-can-be-cited",
          title: 'a post that can be cited',
          description: "9999this is what a post that can be cited looks like",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-04-28-post-citation.html";
            },},{id: "life-a-post-with-typograms",
          title: 'a post with typograms',
          description: "this is what included typograms code could look like",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-04-29-typograms.html";
            },},{id: "life-a-post-with-tabs",
          title: 'a post with tabs',
          description: "this is what included tabs in a post could look like",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-05-01-tabs.html";
            },},{id: "life-222243433",
          title: '222243433',
          description: "2222222",
          section: "Life",handler: () => {
              window.location.href = "/life/2024-12-04-photo-gallery.html";
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